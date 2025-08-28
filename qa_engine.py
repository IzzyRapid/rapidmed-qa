# qa_engine.py
import os, re
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
from rapidfuzz import fuzz

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False


class ProductQASystem:
    """
    Product Q&A engine for Shopify data.
    - Loads from CSV by default, or from Shopify Admin API if SHOPIFY_MODE=api.
    - Fuzzy-matches product by title; can resolve by product URL (/products/<handle>).
    - Extracts attributes (weight, dimensions, battery, flow).
    - Returns a confidence score (0–100).
    """
    def __init__(self, csv_path: str = "products_export.csv"):
        self.csv_path = csv_path
        self.synonyms: Dict[str, str] = {}
        syn_path = os.environ.get("PRODUCT_SYNONYMS_PATH", "synonyms.json")
        if os.path.exists(syn_path):
            try:
                import json
                with open(syn_path, "r") as fh:
                    self.synonyms = json.load(fh)
            except Exception:
                self.synonyms = {}
        self.reload()

    # ---------- Loaders ----------
    def reload(self):
        mode = os.environ.get("SHOPIFY_MODE", "csv").lower()
        if mode == "api":
            self._load_from_api()
        else:
            self._load_from_csv()
        self._prep_dataframe()
        self._build_index()

    def _load_from_csv(self):
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.DataFrame(columns=["Title","Handle","Body (HTML)","Variant Grams","Variant Weight Unit"])

    def _load_from_api(self):
        try:
            from shopify_client import ShopifyClient
            cli = ShopifyClient()
            products = cli.list_products()
            rows = []
            for p in products:
                title = p.get("title","") or ""
                handle = p.get("handle","") or ""
                body_html = p.get("body_html","") or ""
                for v in p.get("variants", []):
                    rows.append({
                        "Title": title,
                        "Handle": handle,
                        "Body (HTML)": body_html,
                        "Variant Grams": v.get("grams"),
                        "Variant Weight Unit": "g",
                        "Variant SKU": v.get("sku"),
                        "Variant Price": v.get("price"),
                    })
            self.df = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=["Title","Handle","Body (HTML)","Variant Grams","Variant Weight Unit"])
        except Exception as e:
            # Keep API failures from killing the app
            print("Shopify API load failed:", e, flush=True)
            self.df = pd.DataFrame(columns=["Title","Handle","Body (HTML)","Variant Grams","Variant Weight Unit"])

    # ---------- Prep ----------
    def _clean_html(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if HAVE_BS4:
            return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
        return re.sub(r"<[^>]+>", " ", text)

    def _prep_dataframe(self) -> None:
        df = self.df
        df['Title'] = df.get('Title', "").fillna('').astype(str)
        df['desc_clean'] = df.get('Body (HTML)', "").fillna('').astype(str).apply(self._clean_html)
        # Optional metafields if your CSV has them
        if 'Specifications (product.metafields.custom.specifications)' in df.columns:
            df['specs_clean'] = df['Specifications (product.metafields.custom.specifications)'].fillna('').astype(str).apply(self._clean_html)
        else:
            df['specs_clean'] = ""
        if 'Key Features (product.metafields.custom.key_features)' in df.columns:
            df['features_clean'] = df['Key Features (product.metafields.custom.key_features)'].fillna('').astype(str).apply(self._clean_html)
        else:
            df['features_clean'] = ""
        self.df = df

    def _build_index(self) -> None:
        self.product_groups: Dict[str, Dict[str, Any]] = {}
        for title, group in self.df.groupby('Title'):
            if not title:
                continue
            info = {
                'handles': list(group.get('Handle', pd.Series(dtype=str)).dropna().astype(str).unique()),
                'desc': " ".join(group['desc_clean'].dropna().astype(str).unique())[:12000],
                'specs': " ".join(group['specs_clean'].dropna().astype(str).unique())[:12000],
                'features': " ".join(group['features_clean'].dropna().astype(str).unique())[:12000],
                'grams': group.get('Variant Grams', pd.Series(dtype=float)).dropna().astype(float).tolist(),
                'weight_unit': group.get('Variant Weight Unit', pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
            }
            self.product_groups[title] = info
        self.titles: List[str] = list(self.product_groups.keys())

    # ---------- Matching ----------
    def resolve_product_from_url(self, q: str) -> Optional[str]:
        m = re.search(r"/products/([a-z0-9\\-]+)", q, flags=re.I)
        if not m:
            return None
        handle = m.group(1).lower()
        for title, info in self.product_groups.items():
            for h in info.get('handles', []):
                if str(h).lower() == handle:
                    return title
        return None

    def _confidence(self, a: str, b: str) -> int:
        try:
            return int(fuzz.partial_ratio(a.lower(), b.lower()))
        except Exception:
            return 0

    def find_best_title(self, query: str, cutoff: float = 0.55) -> Tuple[Optional[str], List[str]]:
        alias = self.synonyms.get(query.lower().strip())
        if alias and alias in self.product_groups:
            return alias, [alias]
        scores = [(t, self._confidence(query, t)) for t in self.titles]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [t for t, s in scores if s >= int(cutoff*100)]
        return (top[0], top[:3]) if top else (None, [])

    # ---------- Attribute extractors ----------
    def _extract_weight(self, info: Dict[str, Any]) -> Optional[str]:
        grams = [g for g in info.get('grams', []) if isinstance(g, (int, float)) and g > 0]
        if grams:
            kg = min(grams) / 1000.0
            return f\"{kg:.1f} kg (approx., from variant data)\"
        text = \" \".join([info.get('specs',''), info.get('features',''), info.get('desc','')])
        m = re.search(r\"(?:weight[^A-Za-z0-9]{0,10})?(\\d{1,3}(?:\\.\\d{1,2})?)\\s?(kg|kilograms|g|grams|lbs|lb|pounds)\", text, flags=re.I)
        if m:
            val, unit = m.groups()
            val = float(val)
            unit = unit.lower()
            if unit in ['kg','kilograms']:
                return f\"{val} kg (from product text)\"
            if unit in ['g','grams']:
                return f\"{val/1000:.1f} kg (from product text)\"
            if unit in ['lbs','lb','pounds']:
                return f\"{val*0.453592:.1f} kg (converted from {unit}; from product text)\"
        return None

    def _extract_dimensions(self, info: Dict[str, Any]) -> Optional[str]:
        text = \" \".join([info.get('specs',''), info.get('features',''), info.get('desc','')])
        m = re.search(r\"(\\d{2,4}(?:\\.\\d{1,2})?)\\s*[x×]\\s*(\\d{2,4}(?:\\.\\d{1,2})?)\\s*[x×]\\s*(\\d{2,4}(?:\\.\\d{1,2})?)\\s*(mm|cm|in|inch|inches)\", text, flags=re.I)
        if m:
            l,w,h,unit = m.groups()
            return f\"{l} x {w} x {h} {unit} (from product text)\"
        m2 = re.search(r\"dimensions[^:]*:\\s*([^\\n\\r;]+)\", text, flags=re.I)
        if m2:
            return f\"{m2.group(1).strip()} (from product text)\"
        return None

    def _extract_battery(self, info: Dict[str, Any]) -> Optional[str]:
        text = \" \".join([info.get('specs',''), info.get('features',''), info.get('desc','')])
        m = re.search(r\"(battery(?:\\s*life|\\s*run[\\s-]*time)?)\\D{0,12}(\\d{1,2}(?:\\.\\d{1,2})?)\\s*(h|hr|hrs|hour|hours)\", text, flags=re.I)
        if m:
            return f\"{m.group(2)} hours (from product text)\"
        return None

    def _extract_flow(self, info: Dict[str, Any]) -> Optional[str]:
        text = \" \".join([info.get('specs',''), info.get('features',''), info.get('desc','')]).lower()
        tags = []
        if 'continuous flow' in text or 'continuous-flow' in text:
            tags.append('continuous flow')
        if 'pulse flow' in text or 'pulse-dose' in text or 'pulse mode' in text:
            tags.append('pulse flow')
        m = re.search(r\"(\\d(?:\\.\\d)?)\\s*(l\\/?min|lpm|litres per minute)\", text, flags=re.I)
        rate = f\"{m.group(1)} {m.group(2).upper()}\" if m else None
        if tags or rate:
            return (\", \".join(tags) + (f\" ({rate})\" if rate else \"\")) if tags else rate
        return None

    # ---------- Answer ----------
    def answer(self, question: str) -> Dict[str, Any]:
        q = question.strip()
        q_l = q.lower()

        # URL handle first
        product_title = self.resolve_product_from_url(q)
        suggestions: List[str] = []
        if not product_title:
            product_title, suggestions = self.find_best_title(q)

        if not product_title:
            return {
                \"ok\": False,
                \"answer\": \"I couldn’t find that product. Please share the exact product name or a link to the product page.\",
                \"suggestions\": suggestions,
                \"confidence\": 0
            }

        info = self.product_groups[product_title]
        # attribute guess
        attr = \"general\"
        if any(k in q_l for k in ['weight','weigh','kg','grams']):
            attr = \"weight\"
        elif any(k in q_l for k in ['dimension','size','width','height','length','depth','folded']):
            attr = \"dimensions\"
        elif any(k in q_l for k in ['battery','runtime','run time','hours']):
            attr = \"battery\"
        elif any(k in q_l for k in ['flow','l/min','lpm','litres per minute','continuous','pulse']):
            attr = \"flow\"

        extractors = {
            \"weight\": self._extract_weight,
            \"dimensions\": self._extract_dimensions,
            \"battery\": self._extract_battery,
            \"flow\": self._extract_flow
        }

        if attr in extractors:
            val = extractors[attr](info)
            conf = self._confidence(q, product_title)
            if val:
                return {\"ok\": True, \"product\": product_title, \"attribute\": attr, \"value\": val,
                        \"confidence\": conf,
                        \"answer\": f\"{attr.capitalize()} for **{product_title}**: {val}\"}
            else:
                return {\"ok\": False, \"product\": product_title, \"attribute\": attr, \"confidence\": conf,
                        \"answer\": f\"I couldn’t find {attr} details for **{product_title}** in the current data.\"}

        # general fallback
        short = info.get('desc','')[:300]
        if len(info.get('desc','')) > 300:
            short += \"...\"
        conf = self._confidence(q, product_title)
        return {\"ok\": True, \"product\": product_title, \"attribute\": \"general\", \"confidence\": conf,
                \"answer\": f\"I found **{product_title}**. Here’s a quick overview: {short}\"}
