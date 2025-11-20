import os
from datetime import datetime, date, timedelta
from calendar import monthrange
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional ObjectId import (avoid hard crash if bson isn't available)
try:
    from bson import ObjectId as _ObjectId
except Exception:
    _ObjectId = None

from database import db, create_document

app = FastAPI(title="Budget Buddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utils ----------

def serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    doc = dict(doc)
    # Prefer stable client_id, else fallback to Mongo _id
    cid = doc.get("client_id")
    if cid:
        doc["id"] = cid
    elif doc.get("_id"):
        try:
            doc["id"] = str(doc.get("_id"))
        except Exception:
            pass
        doc.pop("_id", None)
    # Convert datetime/date to isoformat
    for k, v in list(doc.items()):
        if isinstance(v, (datetime, date)):
            doc[k] = v.isoformat()
    return doc


def month_key(d: Optional[date] = None) -> str:
    d = d or date.today()
    return d.strftime("%Y-%m")


def parse_month(m: Optional[str]) -> str:
    if not m:
        return month_key()
    try:
        datetime.strptime(m, "%Y-%m")
        return m
    except ValueError:
        raise HTTPException(status_code=400, detail="month must be YYYY-MM")


def month_date_range(mkey: str):
    y, m = [int(x) for x in mkey.split("-")]
    start = date(y, m, 1)
    end = date(y, m, monthrange(y, m)[1])
    return start, end


# Build a query that works even if bson isn't available
def id_query(item_id: str) -> Dict[str, Any]:
    q: Dict[str, Any] = {"$or": [{"client_id": item_id}]}
    if _ObjectId is not None:
        try:
            q["$or"].append({"_id": _ObjectId(item_id)})
        except Exception:
            pass
    return q


# ---------- Models ----------

class BudgetSetRequest(BaseModel):
    month: Optional[str] = Field(None, description="YYYY-MM; defaults to current month")
    limit_amount: float = Field(..., ge=0)


class TransactionCreate(BaseModel):
    date: date
    amount: float = Field(..., gt=0)
    merchant: str
    category: Optional[str] = None
    source: str = Field("manual")
    note: Optional[str] = None


class WishlistCreate(BaseModel):
    name: str
    estimated_price: float = Field(..., ge=0)
    priority: str = Field("Medium")
    deadline_month: Optional[str] = None


class WishlistUpdate(BaseModel):
    name: Optional[str] = None
    estimated_price: Optional[float] = Field(None, ge=0)
    priority: Optional[str] = None
    deadline_month: Optional[str] = None
    purchased: Optional[bool] = None


# ---------- Root & Health ----------

@app.get("/")
def root():
    return {"message": "Budget Buddy API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
            response["database"] = "✅ Connected & Working"
    except Exception as e:
        response["database"] = f"⚠️ Connected but error: {str(e)[:80]}"
    return response


# ---------- Budget ----------

@app.get("/api/budget")
def get_budget(month: Optional[str] = Query(None)):
    mkey = parse_month(month)
    if db is None:
        return {"month": mkey, "limit_amount": 0}
    doc = db["budget"].find_one({"month": mkey})
    if not doc:
        return {"month": mkey, "limit_amount": 0}
    return serialize(doc)


@app.post("/api/budget")
def set_budget(payload: BudgetSetRequest):
    mkey = parse_month(payload.month)
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    existing = db["budget"].find_one({"month": mkey})
    if existing:
        db["budget"].update_one({"_id": existing.get("_id")}, {"$set": {"limit_amount": payload.limit_amount, "updated_at": datetime.utcnow()}})
        existing["limit_amount"] = payload.limit_amount
        return serialize(existing)
    # Insert with create_document and attach client_id for durability
    _id = create_document("budget", {"month": mkey, "limit_amount": payload.limit_amount, "client_id": None})
    # Use returned id as client_id
    db["budget"].update_one({"_id": id_query(_id)["$or"][1]["_id"]} if _ObjectId else {"_id": None}, {"$set": {"client_id": _id}})
    return {"id": _id, "month": mkey, "limit_amount": payload.limit_amount}


# ---------- Transactions ----------

@app.get("/api/transactions")
def list_transactions(month: Optional[str] = Query(None)):
    mkey = parse_month(month)
    if db is None:
        return []
    start, end = month_date_range(mkey)
    items = db["transaction"].find({
        "date": {"$gte": datetime.combine(start, datetime.min.time()), "$lte": datetime.combine(end, datetime.max.time())}
    }).sort("date", -1)
    return [serialize(x) for x in items]


@app.post("/api/transactions")
def add_transaction(payload: TransactionCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    data = payload.model_dump()
    _id = create_document("transaction", {**data, "client_id": None})
    # Stamp client_id
    if _ObjectId is not None:
        try:
            db["transaction"].update_one({"_id": _ObjectId(_id)}, {"$set": {"client_id": _id}})
        except Exception:
            pass
    return serialize({"client_id": _id, **data})


@app.post("/api/transactions/upload")
async def upload_statement(file: UploadFile = File(...)):
    # Expect CSV with headers: date,amount,merchant,category
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV supported in this demo")
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    content = (await file.read()).decode("utf-8", errors="ignore")
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Empty file")
    import csv
    reader = csv.DictReader(lines)
    inserted = 0
    for row in reader:
        try:
            dt = row.get("date") or row.get("Date")
            amt = row.get("amount") or row.get("Amount")
            merchant = row.get("merchant") or row.get("Merchant") or "Unknown"
            category = row.get("category") or row.get("Category")
            if not dt or not amt:
                continue
            # try multiple date formats
            parsed = None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(dt, fmt)
                    break
                except Exception:
                    pass
            if not parsed:
                continue
            amount = float(amt)
            _id = create_document("transaction", {
                "date": parsed,
                "amount": amount,
                "merchant": merchant,
                "category": category,
                "source": "statement",
                "client_id": None
            })
            if _ObjectId is not None:
                try:
                    db["transaction"].update_one({"_id": _ObjectId(_id)}, {"$set": {"client_id": _id}})
                except Exception:
                    pass
            inserted += 1
        except Exception:
            continue
    return {"inserted": inserted}


# Placeholder Google Pay webhook (if direct integration available)
@app.post("/api/integrations/gpay/webhook")
async def gpay_webhook(payload: Dict[str, Any]):
    # Expect normalized events list with fields matching TransactionCreate
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    events: List[Dict[str, Any]] = payload.get("events", [])
    count = 0
    for e in events:
        try:
            d = e.get("date")
            if isinstance(d, str):
                try:
                    d = datetime.fromisoformat(d)
                except Exception:
                    d = datetime.utcnow()
            _id = create_document("transaction", {
                "date": d,
                "amount": float(e.get("amount", 0)),
                "merchant": e.get("merchant", "Unknown"),
                "category": e.get("category"),
                "source": "gpay",
                "client_id": None
            })
            if _ObjectId is not None:
                try:
                    db["transaction"].update_one({"_id": _ObjectId(_id)}, {"$set": {"client_id": _id}})
                except Exception:
                    pass
            count += 1
        except Exception:
            continue
    return {"inserted": count}


# ---------- Wishlist ----------

@app.get("/api/wishlist")
def list_wishlist():
    if db is None:
        return []
    items = db["wishlistitem"].find({}).sort([("purchased", 1), ("priority", 1)])
    return [serialize(x) for x in items]


@app.post("/api/wishlist")
def create_wishlist(payload: WishlistCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    data = payload.model_dump()
    _id = create_document("wishlistitem", {**data, "purchased": False, "client_id": None})
    if _ObjectId is not None:
        try:
            db["wishlistitem"].update_one({"_id": _ObjectId(_id)}, {"$set": {"client_id": _id}})
        except Exception:
            pass
    return serialize({"client_id": _id, **data, "purchased": False})


@app.patch("/api/wishlist/{item_id}")
def update_wishlist(item_id: str, payload: WishlistUpdate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    if not updates:
        return {"updated": 0}
    res = db["wishlistitem"].update_one(id_query(item_id), {"$set": {**updates, "updated_at": datetime.utcnow()}})
    return {"updated": res.modified_count}


@app.delete("/api/wishlist/{item_id}")
def delete_wishlist(item_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    res = db["wishlistitem"].delete_one(id_query(item_id))
    return {"deleted": res.deleted_count}


# ---------- Summary & AI Suggestions ----------

@app.get("/api/summary")
def summary(month: Optional[str] = Query(None)):
    mkey = parse_month(month)
    if db is None:
        return {"month": mkey, "limit_amount": 0, "spent": 0, "remaining": 0, "percent_used": 0}
    start, end = month_date_range(mkey)
    agg = list(db["transaction"].aggregate([
        {"$match": {"date": {"$gte": datetime.combine(start, datetime.min.time()), "$lte": datetime.combine(end, datetime.max.time())}}},
        {"$group": {"_id": None, "spent": {"$sum": "$amount"}}}
    ]))
    spent = float(agg[0]["spent"]) if agg else 0.0
    bud = db["budget"].find_one({"month": mkey})
    limit_amount = float(bud.get("limit_amount", 0)) if bud else 0.0
    remaining = max(limit_amount - spent, 0.0)
    percent_used = (spent / limit_amount * 100) if limit_amount > 0 else 0.0
    return {"month": mkey, "limit_amount": limit_amount, "spent": spent, "remaining": remaining, "percent_used": round(percent_used, 2)}


@app.get("/api/suggestions")
def suggestions(month: Optional[str] = Query(None)):
    mkey = parse_month(month)
    start, end = month_date_range(mkey)
    today = date.today()
    # spent
    sum_resp = summary(mkey)
    remaining = sum_resp["remaining"]
    # remaining days including today
    remaining_days = max((end - max(today, start)).days + 1, 1)
    daily_allowance = remaining / remaining_days if remaining_days > 0 else 0

    # wishlist
    if db is None:
        items: List[Dict[str, Any]] = []
    else:
        items = [serialize(x) for x in db["wishlistitem"].find({"purchased": {"$ne": True}})]
    affordable = [i for i in items if i.get("estimated_price", 0) <= remaining]
    affordable_sorted = sorted(affordable, key=lambda x: (x.get("priority", "Medium") != "High", x.get("estimated_price", 0)))

    plan: List[str] = []
    if affordable_sorted:
        pick = affordable_sorted[:2]
        pick_cost = sum([i.get("estimated_price", 0) for i in pick])
        if pick_cost <= remaining:
            plan.append(f"You can buy {', '.join([i['name'] for i in pick])} this month.")
        else:
            plan.append(f"One item fits now: {pick[0]['name']}.")
    else:
        plan.append("No wishlist items fit the current remaining budget. Consider saving for next month.")

    guidance = f"Spend about {daily_allowance:.2f} per day for the remaining {remaining_days} days to stay within budget."

    return {
        "month": mkey,
        "remaining": remaining,
        "remaining_days": remaining_days,
        "daily_allowance": round(daily_allowance, 2),
        "affordable": affordable_sorted,
        "plan": plan,
        "message": guidance
    }


@app.get("/api/alerts")
def alerts(month: Optional[str] = Query(None)):
    mkey = parse_month(month)
    sum_resp = summary(mkey)
    remaining = sum_resp["remaining"]
    alerts: List[str] = []

    # Cheapest wishlist item affordability
    wishlist: List[Dict[str, Any]] = []
    if db is not None:
        wishlist = list(db["wishlistitem"].find({"purchased": {"$ne": True}}))
    if wishlist:
        cheapest = min([w.get("estimated_price", 0) for w in wishlist])
        if remaining < cheapest:
            alerts.append("Remaining budget is below the cost of the cheapest wishlist item.")
        else:
            high_now = [w for w in wishlist if w.get("priority") == "High" and w.get("estimated_price", 0) <= remaining]
            if high_now:
                alerts.append("A high-priority item is affordable now.")

    # Overspend risk: if daily spend last 7 days > daily allowance
    if db is not None:
        today = date.today()
        start7 = datetime.combine(today, datetime.min.time()) - timedelta(days=6)
        agg7 = list(db["transaction"].aggregate([
            {"$match": {"date": {"$gte": start7}}},
            {"$group": {"_id": None, "spent": {"$sum": "$amount"}}}
        ]))
        last7 = float(agg7[0]["spent"]) if agg7 else 0.0

        # daily allowance
        suggestions_resp = suggestions(mkey)
        if suggestions_resp["daily_allowance"] and (last7 / 7.0) > suggestions_resp["daily_allowance"]:
            alerts.append("Spending pace suggests you may exceed the monthly limit.")

    return {"month": mkey, "alerts": alerts}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
