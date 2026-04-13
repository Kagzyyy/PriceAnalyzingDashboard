"""
NSE symbol registry.
Primary source: embedded curated list of ~500 liquid NSE stocks.
Extended: fetch full symbol list from NSE bhavcopy on first run.
"""
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from backend.config import SYMBOLS_PATH

logger = logging.getLogger(__name__)

# Curated base list — NIFTY 500 + key mid-caps with sector mapping
BASE_SYMBOLS = [
    # BANKING & FINANCE
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "AXISBANK", "name": "Axis Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "Banking", "industry": "Public Bank"},
    {"symbol": "BANKBARODA", "name": "Bank of Baroda", "sector": "Banking", "industry": "Public Bank"},
    {"symbol": "PNB", "name": "Punjab National Bank", "sector": "Banking", "industry": "Public Bank"},
    {"symbol": "CANBK", "name": "Canara Bank", "sector": "Banking", "industry": "Public Bank"},
    {"symbol": "UNIONBANK", "name": "Union Bank", "sector": "Banking", "industry": "Public Bank"},
    {"symbol": "IDFCFIRSTB", "name": "IDFC First Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "FEDERALBNK", "name": "Federal Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "INDUSINDBK", "name": "IndusInd Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "YESBANK", "name": "Yes Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "RBLBANK", "name": "RBL Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "BANDHANBNK", "name": "Bandhan Bank", "sector": "Banking", "industry": "Private Bank"},
    {"symbol": "AUBANK", "name": "AU Small Finance Bank", "sector": "Banking", "industry": "Small Finance Bank"},
    {"symbol": "HDFCAMC", "name": "HDFC AMC", "sector": "Finance", "industry": "Asset Management"},
    {"symbol": "NIPPONLIFE", "name": "Nippon Life AMC", "sector": "Finance", "industry": "Asset Management"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "sector": "Finance", "industry": "NBFC"},
    {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv", "sector": "Finance", "industry": "NBFC"},
    {"symbol": "CHOLAFIN", "name": "Cholamandalam Finance", "sector": "Finance", "industry": "NBFC"},
    {"symbol": "LICHSGFIN", "name": "LIC Housing Finance", "sector": "Finance", "industry": "Housing Finance"},
    {"symbol": "PNBHOUSING", "name": "PNB Housing Finance", "sector": "Finance", "industry": "Housing Finance"},
    {"symbol": "MUTHOOTFIN", "name": "Muthoot Finance", "sector": "Finance", "industry": "Gold Loan"},
    {"symbol": "MANAPPURAM", "name": "Manappuram Finance", "sector": "Finance", "industry": "Gold Loan"},
    {"symbol": "SBICARD", "name": "SBI Cards", "sector": "Finance", "industry": "Credit Card"},
    {"symbol": "HDFCLIFE", "name": "HDFC Life Insurance", "sector": "Insurance", "industry": "Life Insurance"},
    {"symbol": "SBILIFE", "name": "SBI Life Insurance", "sector": "Insurance", "industry": "Life Insurance"},
    {"symbol": "ICICIGI", "name": "ICICI Lombard", "sector": "Insurance", "industry": "General Insurance"},
    {"symbol": "ICICIPRULI", "name": "ICICI Prudential Life", "sector": "Insurance", "industry": "Life Insurance"},
    # IT
    {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "IT", "industry": "IT Services"},
    {"symbol": "INFY", "name": "Infosys", "sector": "IT", "industry": "IT Services"},
    {"symbol": "WIPRO", "name": "Wipro", "sector": "IT", "industry": "IT Services"},
    {"symbol": "HCLTECH", "name": "HCL Technologies", "sector": "IT", "industry": "IT Services"},
    {"symbol": "TECHM", "name": "Tech Mahindra", "sector": "IT", "industry": "IT Services"},
    {"symbol": "LTI", "name": "LTIMindtree", "sector": "IT", "industry": "IT Services"},
    {"symbol": "LTIM", "name": "LTIMindtree", "sector": "IT", "industry": "IT Services"},
    {"symbol": "MPHASIS", "name": "Mphasis", "sector": "IT", "industry": "IT Services"},
    {"symbol": "COFORGE", "name": "Coforge", "sector": "IT", "industry": "IT Services"},
    {"symbol": "PERSISTENT", "name": "Persistent Systems", "sector": "IT", "industry": "IT Services"},
    {"symbol": "OFSS", "name": "Oracle Financial Services", "sector": "IT", "industry": "IT Products"},
    {"symbol": "HEXAWARE", "name": "Hexaware Technologies", "sector": "IT", "industry": "IT Services"},
    {"symbol": "KPITTECH", "name": "KPIT Technologies", "sector": "IT", "industry": "IT Services"},
    {"symbol": "TATAELXSI", "name": "Tata Elxsi", "sector": "IT", "industry": "IT Services"},
    {"symbol": "CYIENT", "name": "Cyient", "sector": "IT", "industry": "IT Services"},
    # OIL & GAS / ENERGY
    {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "Energy", "industry": "Oil & Gas"},
    {"symbol": "ONGC", "name": "ONGC", "sector": "Energy", "industry": "Oil & Gas"},
    {"symbol": "BPCL", "name": "BPCL", "sector": "Energy", "industry": "Oil Refining"},
    {"symbol": "IOC", "name": "Indian Oil Corporation", "sector": "Energy", "industry": "Oil Refining"},
    {"symbol": "HPCL", "name": "HPCL", "sector": "Energy", "industry": "Oil Refining"},
    {"symbol": "GAIL", "name": "GAIL India", "sector": "Energy", "industry": "Gas Utilities"},
    {"symbol": "OIL", "name": "Oil India", "sector": "Energy", "industry": "Oil & Gas"},
    {"symbol": "CASTROLIND", "name": "Castrol India", "sector": "Energy", "industry": "Lubricants"},
    {"symbol": "ADANIGREEN", "name": "Adani Green Energy", "sector": "Energy", "industry": "Renewable Energy"},
    {"symbol": "TATAPOWER", "name": "Tata Power", "sector": "Energy", "industry": "Power Generation"},
    {"symbol": "NTPC", "name": "NTPC", "sector": "Energy", "industry": "Power Generation"},
    {"symbol": "POWERGRID", "name": "Power Grid Corp", "sector": "Energy", "industry": "Power Transmission"},
    {"symbol": "ADANIPOWER", "name": "Adani Power", "sector": "Energy", "industry": "Power Generation"},
    {"symbol": "TORNTPOWER", "name": "Torrent Power", "sector": "Energy", "industry": "Power Generation"},
    {"symbol": "CESC", "name": "CESC", "sector": "Energy", "industry": "Power Distribution"},
    {"symbol": "NHPC", "name": "NHPC", "sector": "Energy", "industry": "Hydro Power"},
    # FMCG
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "ITC", "name": "ITC", "sector": "FMCG", "industry": "Tobacco & FMCG"},
    {"symbol": "NESTLEIND", "name": "Nestle India", "sector": "FMCG", "industry": "Food"},
    {"symbol": "BRITANNIA", "name": "Britannia Industries", "sector": "FMCG", "industry": "Food"},
    {"symbol": "DABUR", "name": "Dabur India", "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "MARICO", "name": "Marico", "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "GODREJCP", "name": "Godrej Consumer Products", "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "COLPAL", "name": "Colgate-Palmolive", "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "EMAMILTD", "name": "Emami", "sector": "FMCG", "industry": "Personal Care"},
    {"symbol": "VBL", "name": "Varun Beverages", "sector": "FMCG", "industry": "Beverages"},
    {"symbol": "UBL", "name": "United Breweries", "sector": "FMCG", "industry": "Beverages"},
    {"symbol": "UNITDSPR", "name": "United Spirits", "sector": "FMCG", "industry": "Beverages"},
    {"symbol": "TATACONSUM", "name": "Tata Consumer Products", "sector": "FMCG", "industry": "Food & Beverages"},
    {"symbol": "PATANJALI", "name": "Patanjali Foods", "sector": "FMCG", "industry": "Food"},
    # PHARMA
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "DRREDDY", "name": "Dr Reddy's Laboratories", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "CIPLA", "name": "Cipla", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "DIVISLAB", "name": "Divi's Laboratories", "sector": "Pharma", "industry": "API"},
    {"symbol": "BIOCON", "name": "Biocon", "sector": "Pharma", "industry": "Biologics"},
    {"symbol": "LUPIN", "name": "Lupin", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "AUROPHARMA", "name": "Aurobindo Pharma", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "ALKEM", "name": "Alkem Laboratories", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "TORNTPHARM", "name": "Torrent Pharmaceuticals", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "IPCALAB", "name": "IPCA Laboratories", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "GLENMARK", "name": "Glenmark Pharmaceuticals", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "GRANULES", "name": "Granules India", "sector": "Pharma", "industry": "API"},
    {"symbol": "NATCOPHARM", "name": "Natco Pharma", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "ABBOTINDIA", "name": "Abbott India", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "PFIZER", "name": "Pfizer India", "sector": "Pharma", "industry": "Pharmaceuticals"},
    # AUTO
    {"symbol": "MARUTI", "name": "Maruti Suzuki", "sector": "Auto", "industry": "Passenger Vehicles"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors", "sector": "Auto", "industry": "Automobiles"},
    {"symbol": "M&M", "name": "Mahindra & Mahindra", "sector": "Auto", "industry": "Automobiles"},
    {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto", "sector": "Auto", "industry": "Two Wheelers"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp", "sector": "Auto", "industry": "Two Wheelers"},
    {"symbol": "EICHERMOT", "name": "Eicher Motors", "sector": "Auto", "industry": "Two Wheelers"},
    {"symbol": "TVSMOTORS", "name": "TVS Motor", "sector": "Auto", "industry": "Two Wheelers"},
    {"symbol": "ASHOKLEY", "name": "Ashok Leyland", "sector": "Auto", "industry": "Commercial Vehicles"},
    {"symbol": "TVSMOTOR", "name": "TVS Motor", "sector": "Auto", "industry": "Two Wheelers"},
    {"symbol": "BOSCHLTD", "name": "Bosch", "sector": "Auto", "industry": "Auto Ancillaries"},
    {"symbol": "MOTHERSON", "name": "Samvardhana Motherson", "sector": "Auto", "industry": "Auto Ancillaries"},
    {"symbol": "BHARATFORG", "name": "Bharat Forge", "sector": "Auto", "industry": "Auto Ancillaries"},
    {"symbol": "MINDAIND", "name": "Minda Industries", "sector": "Auto", "industry": "Auto Ancillaries"},
    {"symbol": "EXIDEIND", "name": "Exide Industries", "sector": "Auto", "industry": "Batteries"},
    {"symbol": "AMARAJABAT", "name": "Amara Raja Energy", "sector": "Auto", "industry": "Batteries"},
    # METALS & MINING
    {"symbol": "TATASTEEL", "name": "Tata Steel", "sector": "Metals", "industry": "Steel"},
    {"symbol": "JSWSTEEL", "name": "JSW Steel", "sector": "Metals", "industry": "Steel"},
    {"symbol": "SAIL", "name": "Steel Authority of India", "sector": "Metals", "industry": "Steel"},
    {"symbol": "HINDALCO", "name": "Hindalco Industries", "sector": "Metals", "industry": "Aluminium"},
    {"symbol": "VEDL", "name": "Vedanta", "sector": "Metals", "industry": "Diversified Metals"},
    {"symbol": "NMDC", "name": "NMDC", "sector": "Metals", "industry": "Iron Ore"},
    {"symbol": "COALINDIA", "name": "Coal India", "sector": "Metals", "industry": "Coal"},
    {"symbol": "HINDCOPPER", "name": "Hindustan Copper", "sector": "Metals", "industry": "Copper"},
    {"symbol": "NATIONALUM", "name": "National Aluminium", "sector": "Metals", "industry": "Aluminium"},
    {"symbol": "JSWENERGY", "name": "JSW Energy", "sector": "Energy", "industry": "Power Generation"},
    # INFRA & CAPITAL GOODS
    {"symbol": "LT", "name": "Larsen & Toubro", "sector": "Capital Goods", "industry": "Engineering & Construction"},
    {"symbol": "SIEMENS", "name": "Siemens India", "sector": "Capital Goods", "industry": "Electrical Equipment"},
    {"symbol": "ABB", "name": "ABB India", "sector": "Capital Goods", "industry": "Electrical Equipment"},
    {"symbol": "HAVELLS", "name": "Havells India", "sector": "Capital Goods", "industry": "Electrical Equipment"},
    {"symbol": "BHEL", "name": "BHEL", "sector": "Capital Goods", "industry": "Heavy Electrical Equipment"},
    {"symbol": "CUMMINSIND", "name": "Cummins India", "sector": "Capital Goods", "industry": "Industrial Machinery"},
    {"symbol": "THERMAX", "name": "Thermax", "sector": "Capital Goods", "industry": "Energy & Environment"},
    {"symbol": "BEL", "name": "Bharat Electronics", "sector": "Capital Goods", "industry": "Defence Electronics"},
    {"symbol": "HAL", "name": "Hindustan Aeronautics", "sector": "Capital Goods", "industry": "Defence"},
    {"symbol": "COCHINSHIP", "name": "Cochin Shipyard", "sector": "Capital Goods", "industry": "Shipbuilding"},
    # REALTY
    {"symbol": "DLF", "name": "DLF", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "GODREJPROP", "name": "Godrej Properties", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "PRESTIGE", "name": "Prestige Estates", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "OBEROIDEV", "name": "Oberoi Realty", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "PHOENIXLTD", "name": "Phoenix Mills", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "BRIGADE", "name": "Brigade Enterprises", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "SOBHA", "name": "Sobha", "sector": "Realty", "industry": "Real Estate"},
    {"symbol": "MAHLIFE", "name": "Mahindra Lifespace", "sector": "Realty", "industry": "Real Estate"},
    # TELECOM
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "sector": "Telecom", "industry": "Telecom Services"},
    {"symbol": "IDEA", "name": "Vodafone Idea", "sector": "Telecom", "industry": "Telecom Services"},
    {"symbol": "TATACOMM", "name": "Tata Communications", "sector": "Telecom", "industry": "Telecom Services"},
    {"symbol": "INDIAMART", "name": "IndiaMART InterMESH", "sector": "Telecom", "industry": "Internet"},
    # CONSUMER DURABLES
    {"symbol": "TITAN", "name": "Titan Company", "sector": "Consumer Durables", "industry": "Jewellery & Watches"},
    {"symbol": "VOLTAS", "name": "Voltas", "sector": "Consumer Durables", "industry": "Air Conditioning"},
    {"symbol": "BLUESTAR", "name": "Blue Star", "sector": "Consumer Durables", "industry": "Air Conditioning"},
    {"symbol": "WHIRLPOOL", "name": "Whirlpool India", "sector": "Consumer Durables", "industry": "Home Appliances"},
    {"symbol": "VGUARD", "name": "V-Guard Industries", "sector": "Consumer Durables", "industry": "Electrical Equipment"},
    {"symbol": "KAJARIACER", "name": "Kajaria Ceramics", "sector": "Consumer Durables", "industry": "Ceramics"},
    {"symbol": "CERA", "name": "Cera Sanitaryware", "sector": "Consumer Durables", "industry": "Sanitaryware"},
    {"symbol": "DIXON", "name": "Dixon Technologies", "sector": "Consumer Durables", "industry": "Electronics Manufacturing"},
    {"symbol": "AMBER", "name": "Amber Enterprises", "sector": "Consumer Durables", "industry": "AC Components"},
    # CEMENT
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement", "sector": "Cement", "industry": "Cement"},
    {"symbol": "SHREECEM", "name": "Shree Cement", "sector": "Cement", "industry": "Cement"},
    {"symbol": "AMBUJACEM", "name": "Ambuja Cements", "sector": "Cement", "industry": "Cement"},
    {"symbol": "ACC", "name": "ACC", "sector": "Cement", "industry": "Cement"},
    {"symbol": "DALMIACEMEN", "name": "Dalmia Bharat", "sector": "Cement", "industry": "Cement"},
    {"symbol": "JKCEMENT", "name": "JK Cement", "sector": "Cement", "industry": "Cement"},
    {"symbol": "RAMCOCEM", "name": "Ramco Cements", "sector": "Cement", "industry": "Cement"},
    # CHEMICALS
    {"symbol": "PIDILITIND", "name": "Pidilite Industries", "sector": "Chemicals", "industry": "Adhesives"},
    {"symbol": "SRF", "name": "SRF", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "ATUL", "name": "Atul", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "DEEPAKNTR", "name": "Deepak Nitrite", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "NAVINFLUOR", "name": "Navin Fluorine", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "ALKYLAMINE", "name": "Alkyl Amines Chemicals", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "GALAXY", "name": "Galaxy Surfactants", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "FINEORG", "name": "Fine Organic Industries", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    {"symbol": "VINATIORGA", "name": "Vinati Organics", "sector": "Chemicals", "industry": "Specialty Chemicals"},
    # NEW AGE / TECH
    {"symbol": "NYKAA", "name": "FSN E-Commerce (Nykaa)", "sector": "Consumer", "industry": "E-Commerce"},
    {"symbol": "ZOMATO", "name": "Zomato", "sector": "Consumer", "industry": "Food Delivery"},
    {"symbol": "PAYTM", "name": "One97 Communications (Paytm)", "sector": "Finance", "industry": "Fintech"},
    {"symbol": "POLICYBZR", "name": "PB Fintech (PolicyBazaar)", "sector": "Finance", "industry": "Fintech"},
    {"symbol": "DELHIVERY", "name": "Delhivery", "sector": "Logistics", "industry": "Logistics"},
    {"symbol": "CARTRADE", "name": "CarTrade Tech", "sector": "Consumer", "industry": "Auto Marketplace"},
    # LOGISTICS & TRANSPORT
    {"symbol": "ADANIPORTS", "name": "Adani Ports & SEZ", "sector": "Logistics", "industry": "Ports"},
    {"symbol": "CONCOR", "name": "Container Corporation", "sector": "Logistics", "industry": "Logistics"},
    {"symbol": "BLUEDART", "name": "Blue Dart Express", "sector": "Logistics", "industry": "Courier"},
    {"symbol": "MAHINDCIE", "name": "Mahindra CIE Automotive", "sector": "Auto", "industry": "Auto Ancillaries"},
    # HEALTHCARE
    {"symbol": "APOLLOHOSP", "name": "Apollo Hospitals", "sector": "Healthcare", "industry": "Hospitals"},
    {"symbol": "FORTIS", "name": "Fortis Healthcare", "sector": "Healthcare", "industry": "Hospitals"},
    {"symbol": "MAXHEALTH", "name": "Max Healthcare", "sector": "Healthcare", "industry": "Hospitals"},
    {"symbol": "KIMS", "name": "KIMS Hospitals", "sector": "Healthcare", "industry": "Hospitals"},
    {"symbol": "MEDPLUS", "name": "MedPlus Health Services", "sector": "Healthcare", "industry": "Pharmacy Retail"},
    {"symbol": "METROPOLIS", "name": "Metropolis Healthcare", "sector": "Healthcare", "industry": "Diagnostics"},
    {"symbol": "LALPATHLAB", "name": "Dr Lal PathLabs", "sector": "Healthcare", "industry": "Diagnostics"},
    {"symbol": "THYROCARE", "name": "Thyrocare Technologies", "sector": "Healthcare", "industry": "Diagnostics"},
    # DEFENCE
    {"symbol": "MAZAGONDOCK", "name": "Mazagon Dock Shipbuilders", "sector": "Defence", "industry": "Shipbuilding"},
    {"symbol": "GRSE", "name": "Garden Reach Shipbuilders", "sector": "Defence", "industry": "Shipbuilding"},
    {"symbol": "BEML", "name": "BEML", "sector": "Defence", "industry": "Defence Equipment"},
    {"symbol": "PARAS", "name": "Paras Defence & Space", "sector": "Defence", "industry": "Defence Electronics"},
    # DIVERSIFIED / CONGLOMERATES
    {"symbol": "ADANIENT", "name": "Adani Enterprises", "sector": "Conglomerate", "industry": "Diversified"},
    {"symbol": "ADANITRANS", "name": "Adani Transmission", "sector": "Energy", "industry": "Power Transmission"},
    {"symbol": "TATACHEM", "name": "Tata Chemicals", "sector": "Chemicals", "industry": "Basic Chemicals"},
    {"symbol": "TATACOMM", "name": "Tata Communications", "sector": "Telecom", "industry": "Telecom Services"},
    {"symbol": "TRENT", "name": "Trent", "sector": "Consumer", "industry": "Retail"},
    {"symbol": "ZYDUSLIFE", "name": "Zydus Lifesciences", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "MANKIND", "name": "Mankind Pharma", "sector": "Pharma", "industry": "Pharmaceuticals"},
    {"symbol": "LAURUSLABS", "name": "Laurus Labs", "sector": "Pharma", "industry": "API"},
    {"symbol": "AAVAS", "name": "Aavas Financiers", "sector": "Finance", "industry": "Housing Finance"},
    {"symbol": "CANFINHOME", "name": "Can Fin Homes", "sector": "Finance", "industry": "Housing Finance"},
]


def load_symbols() -> list[dict]:
    """Load symbols from file if exists, else use base list."""
    if SYMBOLS_PATH.exists():
        with open(SYMBOLS_PATH) as f:
            return json.load(f)
    return BASE_SYMBOLS


def save_symbols(symbols: list[dict]) -> None:
    SYMBOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYMBOLS_PATH, "w") as f:
        json.dump(symbols, f, indent=2)


def get_tickers() -> list[str]:
    """Return list of yfinance tickers (symbol + .NS)."""
    symbols = load_symbols()
    # Deduplicate
    seen = set()
    tickers = []
    for s in symbols:
        sym = s["symbol"]
        if sym not in seen:
            seen.add(sym)
            tickers.append(f"{sym}.NS")
    return tickers


def get_symbol_map() -> dict[str, dict]:
    """Return dict: yf_ticker -> {symbol, name, sector, industry}"""
    symbols = load_symbols()
    return {f"{s['symbol']}.NS": s for s in symbols}


def fetch_extended_from_nse() -> list[dict]:
    """
    Attempt to fetch full NSE symbol list from NSE bhavcopy.
    Falls back to base list on failure.
    """
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        # NSE CSV has columns: SYMBOL, NAME OF COMPANY, SERIES, etc.
        df = df[df.get("SERIES", df.columns[2]) == "EQ"]
        symbols = []
        for _, row in df.iterrows():
            sym = str(row.iloc[0]).strip()
            name = str(row.iloc[1]).strip()
            symbols.append({"symbol": sym, "name": name, "sector": "Unknown", "industry": "Unknown"})
        logger.info(f"Fetched {len(symbols)} symbols from NSE")
        return symbols
    except Exception as e:
        logger.warning(f"NSE fetch failed: {e}, using base list")
        return BASE_SYMBOLS


if __name__ == "__main__":
    symbols = load_symbols()
    save_symbols(symbols)
    tickers = get_tickers()
    print(f"Loaded {len(tickers)} NSE tickers")
    print("Sample:", tickers[:5])
