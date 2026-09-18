"""Synthetic Beancount ledger generator.

Every training example is grounded in a small, randomly generated but fully
valid ledger.  The ledgers are loaded with the real Beancount v3 loader, so
account names, currencies, tags and so on that end up in the prompt are the
ones the BQL query is actually executed against.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from decimal import Decimal

from beancount import loader
from beancount.core import data


# --------------------------------------------------------------------------
# Catalogs

REGIONS = {
    "US": dict(
        base="USD", foreign=["EUR", "GBP", "CAD", "JPY", "MXN"],
        banks=["Chase", "BankOfAmerica", "Ally", "Wells-Fargo", "Citi", "Schwab"],
        cards=["Amex", "Visa", "Mastercard", "Discover"],
        employers=["Acme Corp", "Initech", "Globex", "Hooli", "Umbrella Inc"],
        landlords=["Sunrise Properties", "Oakwood Apartments", "Maple Street Rentals"],
        groceries=["Whole Foods", "Trader Joe's", "Safeway", "Kroger", "Costco", "Aldi"],
        restaurants=["Chipotle", "Olive Garden", "Joe's Diner", "Sushi Palace", "Taco Bell", "Luigi's Pizzeria"],
        coffee=["Starbucks", "Blue Bottle", "Peet's Coffee", "Dunkin"],
        fuel=["Shell", "Chevron", "Exxon", "BP"],
        transit=["MTA", "BART", "Metro Card"],
        taxi=["Uber", "Lyft", "Yellow Cab"],
        utilities=["ConEd", "PG&E", "City Water", "National Grid"],
        internet=["Comcast", "Verizon Fios", "AT&T"],
        phone=["T-Mobile", "Verizon Wireless", "Mint Mobile"],
        insurance=["State Farm", "Geico", "Blue Cross"],
        health=["CVS Pharmacy", "Walgreens", "Dr. Miller Clinic", "Smile Dental"],
        shopping=["Amazon", "Target", "Best Buy", "IKEA", "Home Depot", "Nike"],
        cash_name="Cash",
    ),
    "UK": dict(
        base="GBP", foreign=["EUR", "USD", "CHF"],
        banks=["Barclays", "Lloyds", "Monzo", "HSBC", "NatWest"],
        cards=["Amex", "Visa", "Mastercard"],
        employers=["Acme Ltd", "Britannia Tech", "Thames Consulting", "Northern Analytics"],
        landlords=["Kensington Lettings", "Green Door Estates"],
        groceries=["Tesco", "Sainsbury's", "Waitrose", "Asda", "Lidl", "Marks & Spencer"],
        restaurants=["Nando's", "Pizza Express", "Wagamama", "The Crown Pub", "Dishoom"],
        coffee=["Costa Coffee", "Pret a Manger", "Caffe Nero"],
        fuel=["Shell", "BP", "Esso"],
        transit=["TfL", "National Rail", "Oyster"],
        taxi=["Uber", "Bolt", "Black Cab"],
        utilities=["British Gas", "Thames Water", "Octopus Energy"],
        internet=["BT Broadband", "Virgin Media", "Sky"],
        phone=["Vodafone", "EE", "O2"],
        insurance=["Aviva", "Direct Line", "Admiral"],
        health=["Boots", "Superdrug", "Harley Street Clinic"],
        shopping=["Amazon", "John Lewis", "Argos", "IKEA", "Primark"],
        cash_name="Cash",
    ),
    "DE": dict(
        base="EUR", foreign=["USD", "GBP", "CHF", "PLN"],
        banks=["Sparkasse", "Commerzbank", "N26", "DKB", "ING"],
        cards=["Visa", "Mastercard", "Amex"],
        employers=["Muster GmbH", "Bergmann AG", "Rheinland Software", "Nordlicht GmbH"],
        landlords=["Hausverwaltung Schmidt", "Wohnbau Nord"],
        groceries=["Edeka", "Rewe", "Lidl", "Aldi", "Kaufland", "dm"],
        restaurants=["Zur Linde", "Trattoria Roma", "Doner Haus", "Brauhaus", "Vapiano"],
        coffee=["Balzac Coffee", "Einstein Kaffee", "Backerei Muller"],
        fuel=["Aral", "Shell", "Jet"],
        transit=["Deutsche Bahn", "BVG", "MVV"],
        taxi=["FreeNow", "Uber", "Taxi Zentrale"],
        utilities=["Stadtwerke", "Vattenfall", "E.ON"],
        internet=["Telekom", "Vodafone", "1&1"],
        phone=["O2", "Telekom Mobil", "Congstar"],
        insurance=["Allianz", "AOK", "HUK"],
        health=["Apotheke am Markt", "Dr. Weber", "Zahnarzt Fischer"],
        shopping=["Amazon", "MediaMarkt", "IKEA", "Zalando", "Obi"],
        cash_name="Bargeld",
    ),
    "CH": dict(
        base="CHF", foreign=["EUR", "USD", "GBP"],
        banks=["UBS", "PostFinance", "Raiffeisen", "ZKB", "Neon"],
        cards=["Visa", "Mastercard", "Cumulus"],
        employers=["Alpina AG", "Helvetia Tech", "Matterhorn Solutions", "Rhein Consulting"],
        landlords=["Immobilien Zurich", "Verwaltung Berner"],
        groceries=["Migros", "Coop", "Denner", "Aldi Suisse", "Lidl"],
        restaurants=["Zeughauskeller", "Sternen Grill", "Hiltl", "Pizzeria Bella", "Kebab Haus"],
        coffee=["Starbucks", "Sprungli Cafe", "Movenpick"],
        fuel=["Avia", "Shell", "Coop Pronto"],
        transit=["SBB", "ZVV", "BLS"],
        taxi=["Uber", "Taxi 444"],
        utilities=["EWZ", "Stadtwerke", "Wasserversorgung"],
        internet=["Swisscom", "Sunrise", "Salt"],
        phone=["Swisscom Mobile", "Sunrise Mobile", "Wingo"],
        insurance=["Helsana", "CSS", "Swica", "Mobiliar"],
        health=["Apotheke Bahnhof", "Dr. Keller", "Zahnarztpraxis Frei"],
        shopping=["Digitec", "Galaxus", "Manor", "IKEA", "Ex Libris"],
        cash_name="Cash",
    ),
    "CA": dict(
        base="CAD", foreign=["USD", "EUR", "GBP"],
        banks=["RBC", "TD", "Scotiabank", "BMO", "Tangerine"],
        cards=["Visa", "Mastercard", "Amex"],
        employers=["Maple Systems", "Northern Lights Inc", "Laurentian Consulting"],
        landlords=["Cedar Property Management", "Harbourfront Rentals"],
        groceries=["Loblaws", "Sobeys", "Metro", "No Frills", "Costco"],
        restaurants=["Tim Hortons", "Swiss Chalet", "The Keg", "Poutine House", "Pizzaiolo"],
        coffee=["Tim Hortons", "Second Cup", "Starbucks"],
        fuel=["Petro-Canada", "Esso", "Shell"],
        transit=["TTC", "STM", "TransLink"],
        taxi=["Uber", "Lyft", "Beck Taxi"],
        utilities=["Hydro One", "Enbridge", "City Water"],
        internet=["Rogers", "Bell", "Shaw"],
        phone=["Telus", "Fido", "Koodo"],
        insurance=["Sun Life", "Intact", "Manulife"],
        health=["Shoppers Drug Mart", "Dr. Tremblay", "Bright Smile Dental"],
        shopping=["Amazon", "Canadian Tire", "Best Buy", "IKEA", "Roots"],
        cash_name="Cash",
    ),
}

STOCKS = [
    ("AAPL", "Apple Inc.", 120, 220), ("MSFT", "Microsoft Corp.", 200, 450),
    ("GOOG", "Alphabet Inc.", 90, 180), ("VTI", "Vanguard Total Stock Market ETF", 180, 280),
    ("VXUS", "Vanguard Total International Stock ETF", 50, 70), ("BND", "Vanguard Total Bond Market ETF", 70, 80),
    ("NESN", "Nestle SA", 90, 120), ("SPY", "SPDR S&P 500 ETF", 350, 550),
    ("IWDA", "iShares Core MSCI World", 65, 100), ("TSLA", "Tesla Inc.", 150, 350),
    ("NVDA", "NVIDIA Corp.", 40, 140), ("VWRL", "Vanguard FTSE All-World", 85, 120),
]

# category -> (paths for the different naming styles, narrations, amount range, payee catalog key)
CATEGORIES = {
    "groceries": (["Food:Groceries", "Groceries"], ["Weekly groceries", "Groceries", "Food shopping", "Supermarket run"], (18, 145), "groceries"),
    "restaurants": (["Food:Restaurants", "Restaurants"], ["Dinner", "Lunch", "Dinner with friends", "Lunch with colleagues", "Takeaway"], (14, 130), "restaurants"),
    "coffee": (["Food:Coffee", "Coffee"], ["Coffee", "Latte", "Coffee and croissant"], (3, 9), "coffee"),
    "fuel": (["Transport:Fuel", "Fuel"], ["Fuel", "Gas fill-up", "Petrol"], (32, 95), "fuel"),
    "transit": (["Transport:Public", "Transit"], ["Monthly pass", "Train ticket", "Bus fare", "Ticket"], (3, 90), "transit"),
    "taxi": (["Transport:Taxi", "Taxi"], ["Ride home", "Airport ride", "Taxi"], (8, 55), "taxi"),
    "rent": (["Housing:Rent", "Rent"], ["Rent {month}"], (900, 2600), "landlords"),
    "utilities": (["Housing:Utilities", "Utilities"], ["Electricity", "Water bill", "Gas and electricity", "Utilities {month}"], (40, 170), "utilities"),
    "internet": (["Housing:Internet", "Internet"], ["Internet {month}", "Broadband"], (30, 90), "internet"),
    "phone": (["Housing:Phone", "Phone"], ["Phone plan", "Mobile {month}"], (15, 80), "phone"),
    "insurance": (["Insurance:Health", "Insurance"], ["Health insurance premium", "Insurance {month}", "Premium"], (80, 420), "insurance"),
    "health": (["Health:Medical", "Health"], ["Prescription", "Doctor visit", "Dentist", "Pharmacy"], (9, 220), "health"),
    "clothing": (["Shopping:Clothing", "Clothing"], ["Jacket", "Shoes", "Clothes", "Winter coat"], (25, 210), "shopping"),
    "electronics": (["Shopping:Electronics", "Electronics"], ["Headphones", "Laptop accessories", "Monitor", "Phone case", "Keyboard"], (20, 950), "shopping"),
    "household": (["Shopping:Household", "Household"], ["Furniture", "Kitchen supplies", "Tools", "Cleaning supplies"], (12, 320), "shopping"),
    "entertainment": (["Entertainment:Leisure", "Entertainment"], ["Cinema", "Concert tickets", "Museum", "Board games", "Video game"], (8, 140), "shopping"),
    "gifts": (["Gifts", "Gifts"], ["Birthday gift", "Wedding gift", "Christmas present"], (15, 160), "shopping"),
    "charity": (["Charity", "Donations"], ["Donation", "Monthly donation", "Year-end donation"], (10, 200), "shopping"),
    "education": (["Education", "Education"], ["Online course", "Books", "Workshop fee", "Conference ticket"], (15, 400), "shopping"),
}
SUBSCRIPTIONS = [
    ("Netflix", 8, 20), ("Spotify", 9, 13), ("iCloud", 1, 10), ("GitHub", 4, 10), ("The Guardian", 6, 15),
    ("Dropbox", 10, 12), ("Audible", 8, 16), ("YouTube Premium", 12, 14), ("NY Times", 4, 25), ("Gym Membership", 25, 60),
]
TRIP_PLACES = ["Lisbon", "Rome", "Tokyo", "Paris", "Berlin", "Vienna", "Barcelona", "Prague", "Athens", "Copenhagen", "Amsterdam", "Lisbon", "Seoul"]
BUSINESS_CLIENTS = ["Northwind", "Contoso", "Fabrikam", "Tailspin", "Wingtip"]
EVENTS = [("location", ["Zurich", "Berlin", "New York", "London", "Toronto", "Lisbon"]), ("employer", ["Acme Corp", "Initech", "Globex"]), ("address", ["12 Elm Street", "4 River Road", "88 Hill Avenue"])]
INSTITUTIONS_META = ["institution", "bank"]


def money(x: float | Decimal, places: int = 2) -> Decimal:
    return Decimal(str(round(float(x), places))).quantize(Decimal(1).scaleb(-places))


def fmt(amount: Decimal) -> str:
    return format(amount, "f")


# --------------------------------------------------------------------------
# Ledger builder

@dataclass
class _Txn:
    date: dt.date
    flag: str
    payee: str | None
    narration: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    # (account, amount-string, posting meta)
    postings: list[tuple[str, str | None, dict[str, str]]] = field(default_factory=list)

    def render(self) -> str:
        head = f'{self.date.isoformat()} {self.flag}'
        if self.payee is not None:
            head += f' "{self.payee}" "{self.narration}"'
        else:
            head += f' "{self.narration}"'
        for t in self.tags:
            head += f" #{t}"
        for l in self.links:
            head += f" ^{l}"
        lines = [head]
        for k, v in self.meta.items():
            lines.append(f'  {k}: "{v}"')
        for account, amount, pmeta in self.postings:
            lines.append(f"  {account}" + (f"  {amount}" if amount else ""))
            for k, v in pmeta.items():
                lines.append(f'    {k}: "{v}"')
        return "\n".join(lines)


class LedgerBuilder:
    """Builds one random but valid ledger."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.region_name = rng.choice(list(REGIONS))
        self.region = REGIONS[self.region_name]
        self.base = self.region["base"]
        self.foreign = rng.sample(self.region["foreign"], k=rng.randint(1, 2))
        self.naming = rng.choice(["plain", "institution"])
        self.cat_style = rng.choice([0, 1])
        self.years = rng.choice([1, 1, 2, 2, 3])
        end_year = rng.choice([2023, 2024, 2025, 2026])
        self.start = dt.date(end_year - self.years + 1, rng.choice([1, 1, 1, 4, 7]), 1)
        end_month = rng.randint(3, 12)
        self.end = (dt.date(end_year, end_month, 28) if end_year < 2026 else dt.date(2026, min(end_month, 8), 28))
        if self.end <= self.start + dt.timedelta(days=200):
            self.end = self.start + dt.timedelta(days=rng.randint(240, 700))
        self.today = self.end + dt.timedelta(days=rng.randint(2, 20))
        self.txns: list[_Txn] = []
        self.directives: list[tuple[dt.date, int, str]] = []  # (date, order, text)
        self.balances: dict[tuple[str, str], Decimal] = {}
        self.accounts: dict[str, dict] = {}
        self.link_counter = 0
        self.inv_counter = 0
        self._build_accounts()

    # -- naming --------------------------------------------------------
    def _inst(self, kind: str) -> str:
        if self.naming == "plain":
            return ""
        return self.inst_names[kind]

    def _build_accounts(self) -> None:
        rng, region = self.rng, self.region
        bank = rng.choice(region["banks"])
        bank2 = rng.choice([b for b in region["banks"] if b != bank])
        card = rng.choice(region["cards"])
        self.inst_names = {"checking": bank, "savings": rng.choice([bank, bank2]), "card": card, "broker": rng.choice(["Schwab", "Fidelity", "Vanguard", "IBKR", "Degiro"]), "foreign": bank2}

        def acct(root: str, kind: str, plain: str, inst_tail: str | None = None) -> str:
            if self.naming == "plain":
                return f"{root}:{plain}"
            inst = self.inst_names[kind].replace(" ", "-")
            return f"{root}:{self.region_name}:{inst}:{inst_tail or plain}"

        self.a_checking = acct("Assets", "checking", "Checking")
        self.a_savings = acct("Assets", "savings", "Savings") if rng.random() < 0.6 else None
        self.a_card = acct("Liabilities", "card", "CreditCard", "Card") if rng.random() < 0.85 else None
        self.a_cash = f"Assets:{region['cash_name']}" if rng.random() < 0.5 else None
        self.a_foreign = None
        self.foreign_ccy = None
        if rng.random() < 0.45:
            self.foreign_ccy = self.foreign[0]
            self.a_foreign = (f"Assets:{self.foreign_ccy}-Account" if self.naming == "plain"
                              else f"Assets:{self.region_name}:{self.inst_names['foreign'].replace(' ', '-')}:{self.foreign_ccy}")
        self.a_equity = "Equity:Opening-Balances"
        self.i_salary = "Income:Salary" if self.naming == "plain" else f"Income:{rng.choice(['Employer', 'Work'])}:Salary"
        self.i_bonus = self.i_salary.replace("Salary", "Bonus")
        self.i_interest = "Income:Interest"
        self.i_other = rng.choice(["Income:Freelance", "Income:Side-Projects", "Income:Consulting"])
        self.e_tax = "Expenses:Taxes:Income" if rng.random() < 0.6 else None
        self.e_social = "Expenses:Taxes:Social" if self.e_tax and rng.random() < 0.5 else None
        self.e_fees = "Expenses:Bank-Fees"

        n_cat = rng.randint(9, len(CATEGORIES))
        must = ["groceries", "rent"] if rng.random() < 0.8 else ["groceries"]
        pool = [c for c in CATEGORIES if c not in must]
        self.categories = must + rng.sample(pool, k=min(n_cat - len(must), len(pool)))
        self.cat_accounts = {c: "Expenses:" + CATEGORIES[c][0][self.cat_style] for c in self.categories}
        self.use_subscriptions = rng.random() < 0.7
        self.a_subscriptions = "Expenses:" + ("Entertainment:Subscriptions" if self.cat_style == 0 else "Subscriptions")
        self.use_travel = rng.random() < 0.55
        self.a_travel = ["Expenses:Travel:Flights", "Expenses:Travel:Hotels", "Expenses:Travel:Other"] if self.cat_style == 0 else ["Expenses:Flights", "Expenses:Hotels", "Expenses:Travel"]

        # Investments
        self.stocks: list[dict] = []
        if rng.random() < 0.55:
            for sym, name, lo, hi in rng.sample(STOCKS, k=rng.randint(1, 3)):
                broker = self.inst_names["broker"].replace(" ", "-")
                a = f"Assets:Investments:{sym}" if self.naming == "plain" else f"Assets:{broker}:{sym}"
                self.stocks.append(dict(sym=sym, name=name, lo=lo, hi=hi, account=a, price=Decimal(str(rng.uniform(lo, hi))), units=Decimal(0),
                                        booking=rng.choice(["FIFO", "LIFO", "STRICT"]),
                                        income=f"Income:Dividends:{sym}" if rng.random() < 0.7 else "Income:Dividends"))
        self.a_broker_cash = None
        if self.stocks:
            broker = self.inst_names["broker"].replace(" ", "-")
            self.a_broker_cash = "Assets:Investments:Cash" if self.naming == "plain" else f"Assets:{broker}:Cash"
        self.i_gains = "Income:Capital-Gains"
        self.a_loan = None
        if rng.random() < 0.2:
            self.a_loan = "Liabilities:Loan" if self.naming == "plain" else f"Liabilities:{self.region_name}:{self.inst_names['checking'].replace(' ', '-')}:Loan"

        self.use_tags = rng.random() < 0.85
        self.use_meta = rng.random() < 0.6

    # -- helpers -------------------------------------------------------
    def _open(self, date: dt.date, account: str, currencies: list[str] | None = None, booking: str | None = None, meta: dict | None = None) -> None:
        line = f"{date.isoformat()} open {account}"
        if currencies:
            line += " " + ",".join(currencies)
        if booking:
            line += f' "{booking}"'
        for k, v in (meta or {}).items():
            line += f'\n  {k}: "{v}"'
        self.directives.append((date, 0, line))
        self.accounts[account] = dict(open=date, close=None)

    def _bal(self, account: str, ccy: str, delta: Decimal) -> None:
        self.balances[(account, ccy)] = self.balances.get((account, ccy), Decimal(0)) + delta

    def _add(self, txn: _Txn) -> None:
        # postings must be fully explicit: (account, Decimal amount, currency, suffix, pmeta)
        self.txns.append(txn)

    def _txn(self, date, payee, narration, legs, *, flag="*", tags=None, links=None, meta=None) -> None:
        """legs: list of (account, Decimal|None, ccy|None, suffix, pmeta)."""
        if date > self.end:
            return
        postings = []
        for account, amount, ccy, suffix, pmeta in legs:
            if amount is None:
                postings.append((account, None, pmeta))
            else:
                postings.append((account, f"{fmt(amount)} {ccy}" + (f" {suffix}" if suffix else ""), pmeta))
                if not suffix or "@" in suffix:
                    self._bal(account, ccy, amount)
                else:  # cost basis suffix: still track units
                    self._bal(account, ccy, amount)
        self.txns.append(_Txn(date, flag, payee, narration, tags or [], links or [], meta or {}, postings))

    def _payment_account(self) -> str:
        r = self.rng.random()
        if self.a_card and r < 0.55:
            return self.a_card
        if self.a_cash and r < 0.65:
            return self.a_cash
        return self.a_checking

    def _spend(self, date, cat, *, amount=None, payee=None, narration=None, flag="*", tags=None, links=None, meta=None, account=None, pay=None) -> None:
        rng = self.rng
        _, narrs, (lo, hi), catalog = CATEGORIES[cat]
        payee = payee or rng.choice(self.region[catalog])
        narration = (narration or rng.choice(narrs)).replace("{month}", date.strftime("%B"))
        amount = amount if amount is not None else money(rng.uniform(lo, hi))
        pay = pay or self._payment_account()
        pmeta = {}
        if self.use_meta and rng.random() < 0.12:
            pmeta = {"category-note": rng.choice(["work", "personal", "shared", "reimbursable"])}
        self._txn(date, payee, narration,
                  [(account or self.cat_accounts[cat], amount, self.base, "", pmeta), (pay, -amount, self.base, "", {})],
                  flag=flag, tags=tags, links=links, meta=meta)

    # -- generation ----------------------------------------------------
    def build(self) -> str:
        rng = self.rng
        start, end = self.start, self.end
        open_date = start - dt.timedelta(days=rng.randint(0, 3))
        meta_inst = self.use_meta and rng.random() < 0.7

        self.directives.append((open_date, -2, f'option "title" "{rng.choice(["Household", "Personal Finances", "Family Ledger", "Main Ledger", "My Books"])}"'))
        self.directives.append((open_date, -2, f'option "operating_currency" "{self.base}"'))

        def om(kind):
            return {"institution": self.inst_names[kind]} if meta_inst and self.naming == "institution" else None

        self._open(open_date, self.a_checking, [self.base], meta=om("checking"))
        if self.a_savings:
            self._open(open_date, self.a_savings, [self.base], meta=om("savings"))
        if self.a_card:
            self._open(open_date, self.a_card, [self.base], meta=om("card"))
        if self.a_cash:
            self._open(open_date, self.a_cash, [self.base])
        if self.a_foreign:
            self._open(open_date, self.a_foreign, [self.foreign_ccy], meta=om("foreign"))
        self._open(open_date, self.a_equity)
        for acc in [self.i_salary, self.i_bonus, self.i_interest, self.i_other]:
            self._open(open_date, acc)
        for acc in [self.e_tax, self.e_social, self.e_fees]:
            if acc:
                self._open(open_date, acc)
        for acc in self.cat_accounts.values():
            self._open(open_date, acc)
        if self.use_subscriptions:
            self._open(open_date, self.a_subscriptions)
        if self.use_travel:
            for acc in self.a_travel:
                self._open(open_date, acc)
        if self.stocks:
            self._open(open_date, self.a_broker_cash, [self.base])
            self._open(open_date, self.i_gains)
            for s in self.stocks:
                self._open(open_date, s["account"], [s["sym"]], booking=s["booking"] if s["booking"] != "STRICT" else None)
                if s["income"] not in self.accounts:
                    self._open(open_date, s["income"])
        if self.a_loan:
            self._open(open_date, self.a_loan, [self.base])
            self._open(open_date, "Expenses:Interest")

        # commodity directives
        commodities = {self.base} | set(self.foreign)
        for c in sorted(commodities):
            meta = {"name": c} if rng.random() < 0.4 else {}
            self.directives.append((open_date, 1, f"{open_date.isoformat()} commodity {c}" + "".join(f'\n  {k}: "{v}"' for k, v in meta.items())))
        for s in self.stocks:
            self.directives.append((open_date, 1, f'{open_date.isoformat()} commodity {s["sym"]}\n  name: "{s["name"]}"\n  asset-class: "{rng.choice(["stock", "etf"])}"'))

        # Opening balances
        chk0 = money(rng.uniform(1500, 9000))
        legs = [(self.a_checking, chk0, self.base, "", {})]
        total = chk0
        if self.a_savings:
            s0 = money(rng.uniform(2000, 30000))
            legs.append((self.a_savings, s0, self.base, "", {}))
            total += s0
        if self.a_cash:
            c0 = money(rng.uniform(50, 400))
            legs.append((self.a_cash, c0, self.base, "", {}))
            total += c0
        legs.append((self.a_equity, -total, self.base, "", {}))
        self._txn(open_date, None, "Opening balances", legs)

        # Monthly loop
        salary = money(rng.uniform(2800, 9500), 0)
        month = dt.date(start.year, start.month, 1)
        trip_no = 0
        self.sub_list = rng.sample(SUBSCRIPTIONS, k=rng.randint(1, 4)) if self.use_subscriptions else []
        self.sub_amounts = {n: money(rng.uniform(lo, hi)) for n, lo, hi in self.sub_list}
        fixed_rent = money(rng.uniform(*CATEGORIES["rent"][2]), 0)
        landlord = rng.choice(self.region["landlords"])
        fixed_fees = {c: money(rng.uniform(*CATEGORIES[c][2])) for c in ("internet", "phone", "insurance", "utilities")}
        fixed_payee = {c: rng.choice(self.region[CATEGORIES[c][3]]) for c in ("internet", "phone", "insurance", "utilities")}
        loan_balance = money(rng.uniform(5000, 20000), 0) if self.a_loan else None
        if self.a_loan:
            self._txn(open_date, "Lender", "Loan disbursement", [(self.a_checking, loan_balance, self.base, "", {}), (self.a_loan, -loan_balance, self.base, "", {})])
        while month <= end:
            def day(n: int) -> dt.date:
                return month + dt.timedelta(days=min(n, 27))
            y, m = month.year, month.month
            tag_month = []
            # salary
            gross = salary
            tax = money(gross * Decimal("0.18")) if self.e_tax else Decimal(0)
            social = money(gross * Decimal("0.06")) if self.e_social else Decimal(0)
            net = gross - tax - social
            legs = [(self.a_checking, net, self.base, "", {})]
            if tax:
                legs.append((self.e_tax, tax, self.base, "", {}))
            if social:
                legs.append((self.e_social, social, self.base, "", {}))
            legs.append((self.i_salary, -gross, self.base, "", {}))
            employer = rng.choice(self.region["employers"]) if month == dt.date(start.year, start.month, 1) else self.employer if hasattr(self, "employer") else None
            if not hasattr(self, "employer"):
                self.employer = employer
            self._txn(day(rng.choice([24, 25, 26, 27])), self.employer, f"Salary {month.strftime('%B %Y')}" if rng.random() < 0.5 else "Salary", legs,
                      tags=["payroll"] if self.use_tags and rng.random() < 0.3 else None)
            if m in (3, 12) and rng.random() < 0.6:
                bonus = money(rng.uniform(500, 6000), 0)
                self._txn(day(27), self.employer, "Bonus", [(self.a_checking, bonus, self.base, "", {}), (self.i_bonus, -bonus, self.base, "", {})])
            if self.a_savings and rng.random() < 0.7:
                interest = money(rng.uniform(1, 45))
                self._txn(day(28), self.inst_names["savings"] if self.naming == "institution" else "Bank", "Interest", [(self.a_savings, interest, self.base, "", {}), (self.i_interest, -interest, self.base, "", {})])
            if self.a_savings and rng.random() < 0.5:
                amt = money(rng.uniform(100, 1200), 0)
                self._txn(day(rng.randint(3, 8)), None, "Transfer to savings", [(self.a_savings, amt, self.base, "", {}), (self.a_checking, -amt, self.base, "", {})])
            # freelance
            if rng.random() < 0.25:
                fee = money(rng.uniform(300, 3500), 0)
                client = rng.choice(BUSINESS_CLIENTS)
                self.link_counter += 1
                meta = {"invoice": f"INV-{y}-{self.link_counter:03d}"} if self.use_meta else {}
                self._txn(day(rng.randint(5, 20)), client, f"Consulting invoice", [(self.a_checking, fee, self.base, "", {}), (self.i_other, -fee, self.base, "", {})],
                          links=[f"invoice-{y}-{self.link_counter:03d}"], tags=["business"] if self.use_tags else None, meta=meta)
            # rent and fixed bills
            if "rent" in self.categories:
                self._spend(day(1), "rent", amount=fixed_rent, payee=landlord, pay=self.a_checking)
            for c in ("internet", "phone", "insurance", "utilities"):
                if c in self.categories:
                    amt = fixed_fees[c] + (money(rng.uniform(-8, 8)) if c == "utilities" else Decimal(0))
                    self._spend(day(rng.randint(4, 12)), c, amount=max(amt, Decimal(5)), payee=fixed_payee[c], pay=self.a_checking)
            for n, _, _ in self.sub_list:
                self._txn(day(rng.randint(2, 18)), n, "Monthly subscription", [(self.a_subscriptions, self.sub_amounts[n], self.base, "", {}), (self._payment_account_fixed(), -self.sub_amounts[n], self.base, "", {})])
            # variable spending
            for c in self.categories:
                if c in ("rent", "internet", "phone", "insurance", "utilities"):
                    continue
                lo_n, hi_n = {"groceries": (4, 9), "restaurants": (2, 6), "coffee": (2, 7), "fuel": (0, 3), "transit": (0, 3), "taxi": (0, 2), "health": (0, 2)}.get(c, (0, 2))
                for _ in range(rng.randint(lo_n, hi_n)):
                    tags = None
                    if c == "restaurants" and self.use_tags and rng.random() < 0.08:
                        tags = ["date-night"]
                    flag = "!" if rng.random() < 0.015 else "*"
                    meta = None
                    if self.use_meta and c in ("electronics", "household", "clothing") and rng.random() < 0.5:
                        meta = {"receipt": f"receipts/{y}/{m:02d}-{rng.randint(100, 999)}.pdf"}
                    tg = list(tags or [])
                    if self.use_tags and c in ("electronics", "education") and rng.random() < 0.3:
                        tg.append("tax-deductible")
                    if self.use_tags and c == "gifts" and rng.random() < 0.6:
                        tg.append("gift")
                    self._spend(day(rng.randint(1, 27)), c, flag=flag, tags=tg or None, meta=meta)
            # bank fees
            if rng.random() < 0.2:
                fee = money(rng.uniform(1, 15))
                self._txn(day(rng.randint(1, 27)), self.inst_names["checking"], "Account fee", [(self.e_fees, fee, self.base, "", {}), (self.a_checking, -fee, self.base, "", {})])
            # refund
            if "electronics" in self.categories and rng.random() < 0.06:
                amt = money(rng.uniform(15, 150))
                self._txn(day(rng.randint(5, 20)), rng.choice(self.region["shopping"]), "Refund", [(self.a_checking, amt, self.base, "", {}), (self.cat_accounts["electronics"], -amt, self.base, "", {})])
            # credit card payment
            if self.a_card:
                owed = -self.balances.get((self.a_card, self.base), Decimal(0))
                if owed > 0:
                    pay = owed if rng.random() < 0.85 else money(owed * Decimal("0.7"))
                    self._txn(day(rng.randint(12, 20)), self.inst_names["card"] if self.naming == "institution" else "Credit Card", "Credit card payment",
                              [(self.a_card, pay, self.base, "", {}), (self.a_checking, -pay, self.base, "", {})])
            # cash withdrawals
            if self.a_cash and rng.random() < 0.5:
                amt = money(rng.uniform(40, 250), 0)
                self._txn(day(rng.randint(1, 26)), None, "ATM withdrawal", [(self.a_cash, amt, self.base, "", {}), (self.a_checking, -amt, self.base, "", {})])
            # foreign account
            if self.a_foreign and rng.random() < 0.3:
                foreign_amt = money(rng.uniform(200, 1500), 0)
                rate = Decimal(str(round(rng.uniform(0.7, 1.4), 3)))
                base_amt = (foreign_amt * rate).quantize(Decimal("0.01"))
                self._txn(day(rng.randint(3, 22)), None, f"Currency exchange {self.base}/{self.foreign_ccy}",
                          [(self.a_foreign, foreign_amt, self.foreign_ccy, f"@@ {fmt(base_amt)} {self.base}", {}), (self.a_checking, -base_amt, self.base, "", {})])
            # travel
            if self.use_travel and rng.random() < 0.18:
                trip_no += 1
                place = rng.choice(TRIP_PLACES)
                tag = f"trip-{place.lower()}-{y}"
                d0 = day(rng.randint(3, 20))
                flight = money(rng.uniform(120, 900), 0)
                hotel = money(rng.uniform(100, 250), 0)
                nights = rng.randint(2, 7)
                pay = self.a_card or self.a_checking
                self._txn(d0 - dt.timedelta(days=rng.randint(10, 40)), rng.choice(["Lufthansa", "Delta", "British Airways", "Swiss", "Ryanair", "KLM"]), f"Flight to {place}",
                          [(self.a_travel[0], flight, self.base, "", {}), (pay, -flight, self.base, "", {})], tags=[tag] if self.use_tags else None)
                if self.foreign_ccy and rng.random() < 0.6:
                    hotel_f = money(hotel * nights)
                    hotel_b = money(hotel_f * Decimal(str(round(rng.uniform(0.7, 1.4), 3))))
                    self._txn(d0, rng.choice(["Hotel Central", "Airbnb", "Booking.com", "Grand Hotel"]), f"Hotel in {place} ({nights} nights)",
                              [(self.a_travel[1], hotel_f, self.foreign_ccy, f"@@ {fmt(hotel_b)} {self.base}", {}), (pay, -hotel_b, self.base, "", {})], tags=[tag] if self.use_tags else None)
                else:
                    total_h = hotel * nights
                    self._txn(d0, rng.choice(["Hotel Central", "Airbnb", "Booking.com", "Grand Hotel"]), f"Hotel in {place} ({nights} nights)",
                              [(self.a_travel[1], total_h, self.base, "", {}), (pay, -total_h, self.base, "", {})], tags=[tag] if self.use_tags else None)
                for _ in range(rng.randint(1, 4)):
                    amt = money(rng.uniform(8, 120))
                    self._txn(d0 + dt.timedelta(days=rng.randint(0, nights)), rng.choice(["Local Restaurant", "Museum Shop", "Metro", "Street Market", "Souvenirs"]), f"{place} expenses",
                              [(self.a_travel[2], amt, self.base, "", {}), (pay, -amt, self.base, "", {})], tags=[tag] if self.use_tags else None)
            # investments
            if self.stocks and self.a_broker_cash:
                if rng.random() < 0.7:
                    dep = money(rng.uniform(300, 2500), 0)
                    self._txn(day(rng.randint(5, 10)), None, "Transfer to brokerage", [(self.a_broker_cash, dep, self.base, "", {}), (self.a_checking, -dep, self.base, "", {})])
                for s in self.stocks:
                    s["price"] = max(Decimal(str(s["lo"] * 0.5)), s["price"] * Decimal(str(round(rng.uniform(0.94, 1.07), 4))))
                    px = s["price"].quantize(Decimal("0.01"))
                    self.directives.append((day(28), 3, f'{day(28).isoformat()} price {s["sym"]} {fmt(px)} {self.base}'))
                    cash = self.balances.get((self.a_broker_cash, self.base), Decimal(0))
                    if rng.random() < 0.45 and cash > px * 2:
                        units = Decimal(rng.randint(1, max(1, min(int(cash // px), 40))))
                        cost = px
                        self._txn(day(rng.randint(10, 20)), self.inst_names["broker"], f"Buy {s['sym']}",
                                  [(s["account"], units, s["sym"], f"{{{fmt(cost)} {self.base}}}", {}), (self.a_broker_cash, -(units * cost), self.base, "", {})])
                        s["units"] += units
                    elif rng.random() < 0.08 and s["units"] > 1:
                        units = Decimal(rng.randint(1, int(s["units"])))
                        if s["booking"] == "STRICT":
                            continue
                        proceeds = units * px
                        self._txn(day(rng.randint(10, 24)), self.inst_names["broker"], f"Sell {s['sym']}",
                                  [(s["account"], -units, s["sym"], f"{{}} @ {fmt(px)} {self.base}", {}), (self.a_broker_cash, proceeds, self.base, "", {}), (self.i_gains, None, None, "", {})])
                        s["units"] -= units
                    if rng.random() < 0.15 and s["units"] > 0:
                        dvd = money(float(s["units"]) * rng.uniform(0.2, 1.4))
                        self._txn(day(rng.randint(14, 24)), s["name"], f"Dividend {s['sym']}", [(self.a_broker_cash, dvd, self.base, "", {}), (s["income"], -dvd, self.base, "", {})])
            # loan
            if self.a_loan and loan_balance and loan_balance > 0:
                pay = min(loan_balance, money(rng.uniform(150, 450), 0))
                interest = money(pay * Decimal("0.08"))
                self._txn(day(rng.randint(8, 15)), "Lender", "Loan payment",
                          [(self.a_loan, pay - interest, self.base, "", {}), ("Expenses:Interest", interest, self.base, "", {}), (self.a_checking, -pay, self.base, "", {})])
                loan_balance -= pay - interest
            # fx price directives
            for f in self.foreign:
                rate = Decimal(str(round(rng.uniform(0.7, 1.4), 4)))
                self.directives.append((day(27), 3, f"{day(27).isoformat()} price {f} {fmt(rate)} {self.base}"))
            # balance assertions at the start of the following month
            nxt = (month + dt.timedelta(days=32)).replace(day=1)
            if nxt <= end + dt.timedelta(days=5) and rng.random() < 0.5:
                pass  # emitted after all txns of the month are known, see below
            month = (month + dt.timedelta(days=32)).replace(day=1)

        # Events / notes / documents
        for ev, values in rng.sample(EVENTS, k=rng.randint(0, 2)):
            for _ in range(rng.randint(1, 3)):
                d = start + dt.timedelta(days=rng.randint(0, (end - start).days))
                self.directives.append((d, 4, f'{d.isoformat()} event "{ev}" "{rng.choice(values)}"'))
        if rng.random() < 0.6:
            for _ in range(rng.randint(1, 3)):
                d = start + dt.timedelta(days=rng.randint(0, (end - start).days))
                self.directives.append((d, 4, f'{d.isoformat()} note {self.a_checking} "{rng.choice(["Called the bank about a fee", "Card replaced", "Changed PIN", "Statement disputed"])}"'))
        if rng.random() < 0.4:
            d = start + dt.timedelta(days=rng.randint(0, (end - start).days))
            self.directives.append((d, 4, f'{d.isoformat()} document {self.a_checking} "statements/{d.year}-{d.month:02d}.pdf"'))
        # Close an account late
        if self.a_cash and rng.random() < 0.2 and self.balances.get((self.a_cash, self.base), 0) == 0:
            pass

        return self._render()

    def _payment_account_fixed(self) -> str:
        return self.a_card if self.a_card and self.rng.random() < 0.6 else self.a_checking

    def _render(self) -> str:
        rng = self.rng
        # balance assertions: emitted on the 1st of some months, using the sim balances at that point
        by_month_end = self._balance_assertions()
        items: list[tuple[dt.date, int, str]] = list(self.directives) + by_month_end
        for t in self.txns:
            items.append((t.date, 2, t.render()))
        items.sort(key=lambda x: (x[0], x[1]))
        return "\n\n".join(text for _, _, text in items) + "\n"

    def _balance_assertions(self) -> list[tuple[dt.date, int, str]]:
        """Replay the transactions and assert balances of the main accounts at some month starts."""
        rng = self.rng
        out: list[tuple[dt.date, int, str]] = []
        running: dict[tuple[str, str], Decimal] = {}
        txns = sorted(self.txns, key=lambda t: t.date)
        watch = [self.a_checking] + ([self.a_savings] if self.a_savings else []) + ([self.a_card] if self.a_card else [])
        idx = 0
        first = dt.date(self.start.year, self.start.month, 1)
        month = (first + dt.timedelta(days=32)).replace(day=1)
        fail_once = rng.random() < 0.15
        while month <= self.end:
            while idx < len(txns) and txns[idx].date < month:
                for account, amount, _ in txns[idx].postings:
                    if amount is None:
                        continue
                    number, ccy = amount.split()[0], amount.split()[1]
                    running[(account, ccy)] = running.get((account, ccy), Decimal(0)) + Decimal(number)
                idx += 1
            if rng.random() < 0.4:
                account = rng.choice(watch)
                value = running.get((account, self.base), Decimal(0))
                if fail_once and rng.random() < 0.3:
                    value += money(rng.uniform(1, 40))
                    fail_once = False
                out.append((month, -1, f"{month.isoformat()} balance {account} {fmt(value)} {self.base}"))
            month = (month + dt.timedelta(days=32)).replace(day=1)
        return out


# --------------------------------------------------------------------------
# Loaded ledger

@dataclass
class Ledger:
    text: str
    entries: list
    errors: list
    options: dict
    today: dt.date
    base: str

    # derived
    accounts: dict = field(default_factory=dict)      # name -> (open Entry, close Entry|None)
    currencies: list = field(default_factory=list)
    commodities: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    links: list = field(default_factory=list)
    payees: list = field(default_factory=list)
    txn_meta_keys: list = field(default_factory=list)
    posting_meta_keys: list = field(default_factory=list)
    first_date: dt.date | None = None
    last_date: dt.date | None = None
    txns: list = field(default_factory=list)

    @classmethod
    def load(cls, text: str, today: dt.date, base: str) -> "Ledger":
        entries, errors, options = loader.load_string(text)
        led = cls(text, entries, errors, options, today, base)
        led._derive()
        return led

    def _derive(self) -> None:
        opens, closes = {}, {}
        currencies, tags, links = {}, {}, {}
        payees: dict[str, int] = {}
        tm, pm = set(), set()
        commodities = []
        for e in self.entries:
            if isinstance(e, data.Open):
                opens[e.account] = e
            elif isinstance(e, data.Close):
                closes[e.account] = e
            elif isinstance(e, data.Commodity):
                commodities.append(e.currency)
            elif isinstance(e, data.Transaction):
                self.txns.append(e)
                for t in e.tags or ():
                    tags[t] = tags.get(t, 0) + 1
                for l in e.links or ():
                    links[l] = links.get(l, 0) + 1
                if e.payee:
                    payees[e.payee] = payees.get(e.payee, 0) + 1
                tm.update(k for k in e.meta if k not in ("filename", "lineno") and not k.startswith("__"))
                for p in e.postings:
                    if p.meta:
                        pm.update(k for k in p.meta if k not in ("filename", "lineno") and not k.startswith("__"))
                    currencies[p.units.currency] = 1
        self.accounts = {a: (o, closes.get(a)) for a, o in opens.items()}
        self.currencies = sorted(currencies)
        self.commodities = sorted(commodities)
        self.tags = sorted(tags)
        self.links = sorted(links)
        self.payees = [p for p, _ in sorted(payees.items(), key=lambda kv: (-kv[1], kv[0]))]
        self.txn_meta_keys = sorted(tm)
        self.posting_meta_keys = sorted(pm)
        dates = [t.date for t in self.txns]
        self.first_date = min(dates) if dates else None
        self.last_date = max(dates) if dates else None

    # ---- helpers for intents
    def accounts_of(self, root: str) -> list[str]:
        return [a for a in self.accounts if a.split(":")[0] == root]

    def used_accounts(self) -> set[str]:
        return {p.account for t in self.txns for p in t.postings}


def generate_ledger(seed: int) -> Ledger:
    """Generate a valid ledger for the given seed (retrying on unexpected loader errors)."""
    attempt = 0
    while True:
        rng = random.Random(seed * 1000 + attempt)
        builder = LedgerBuilder(rng)
        text = builder.build()
        led = Ledger.load(text, builder.today, builder.base)
        bad = [e for e in led.errors if "Balance failed" not in str(getattr(e, "message", e))]
        if not bad and led.txns:
            led.builder = builder  # type: ignore[attr-defined]
            return led
        attempt += 1
        if attempt > 20:
            raise RuntimeError(f"could not generate a valid ledger for seed {seed}: {bad[:3]}")
