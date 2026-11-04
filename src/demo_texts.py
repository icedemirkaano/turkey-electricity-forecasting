"""Texts of the demo app (app/streamlit_app.py) in English and Turkish, with date and number formatting per language.

Every text is looked up by key with `text(lang, key, **values)`; both languages must have the same keys
(tests/test_demo_texts.py). Placeholders use str.format: {cost}, {days}, ...
"""

from datetime import date

LANGUAGES = {"en": "English", "tr": "Türkçe"}

TEXTS = {
    "en": {
        "page_title": "Should the plant run tomorrow?",
        "title": "Should the plant run tomorrow?",
        "intro": "For the operator of a power plant (1 MW, e.g. a gas unit). The auction decides hour by hour whether "
                 "an offer is accepted, but the block to offer, the gas to nominate and the crew must be fixed before "
                 "12:30, without knowing the prices. Prices: Turkish day-ahead market (PTF), USD/MWh. Forecasts by a "
                 "model retrained every Monday, for days no model choice used ({first} – {last}). Data: EPİAŞ "
                 "Transparency Platform; weather data by [Open-Meteo.com](https://open-meteo.com/).",
        "language": "Language",
        "your_plant": "Your plant",
        "cost": "Variable cost (fuel), USD/MWh",
        "min_hours": "Minimum run time, hours",
        "start_cost": "Cost of one start, USD",
        "assumptions": "Assumptions, not a real plant's figures. The plant runs at most one block of consecutive "
                       "hours a day and does not move the price.",
        "tab_plan": "Tomorrow's plan",
        "tab_trust": "Can I trust it?",
        "delivery_day": "Delivery day",
        "forecast_made": "Forecast made on **{day}**, before the 12:30 offer deadline.",
        "show_actual": "Show what actually happened",
        "forecast": "forecast",
        "actual": "actual",
        "your_cost": "your cost {cost}",
        "hour": "hour",
        "plan_chart": "Forecast price per hour (green: block to offer, gray: above cost but outside it, red: below "
                      "cost)",
        "hover_hour": "%{{x}}:00 — %{{y:.1f}} USD/MWh",
        "run": "**Prepare to run: offer {block}.** Expected profit after the start cost: about **{profit} USD** "
               "per MW.",
        "stay_off": "**Stay off tomorrow.** No block of at least {hours} hours is expected to cover the {start} USD "
                    "start cost at a fuel cost of {cost} USD/MWh.",
        "block": "{first:02d}:00–{end:02d}:00 ({hours} hours)",
        "off": "stay off",
        "outcome_title": "How the decision turned out (profit at the actual prices, USD per MW)",
        "follow_model": "Following the model",
        "follow_baseline": "Following same hour yesterday",
        "perfect_best": "Best possible (perfect foresight)",
        "runs": "runs: {block}",
        "day_error": "The forecast's average error on this day: {error} USD/MWh.",
        "trust_intro": "All {days} demo days, each forecast made on the day before and compared with the actual "
                       "prices afterwards. Reference: *same hour yesterday*, a forecast anyone has for free.",
        "model_mae": "Model error (MAE, USD/MWh)",
        "baseline_mae": "Same hour yesterday (MAE)",
        "days_better": "Days the model was better",
        "of": "{wins} of {days}",
        "pays_off": "Does it pay off for your plant? ({days} days, USD per MW)",
        "always_run": "Always run",
        "perfect": "Perfect foresight",
        "gain_earned": "At a fuel cost of {cost} USD/MWh, following the model instead of yesterday's prices earned "
                       "**{amount} USD per MW** over {days} days.",
        "gain_lost": "At a fuel cost of {cost} USD/MWh, following the model instead of yesterday's prices lost "
                     "**{amount} USD per MW** over {days} days.",
        "gain_chart": "Extra profit from the model vs. same hour yesterday, by plant cost (USD per MW)",
        "hover_cost": "cost %{{x}}: %{{y:,.0f}} USD",
        "cost_axis": "variable cost, USD/MWh",
        "gain_note": "The forecast matters least at the ends: a very cheap plant runs anyway, a very expensive one "
                     "never does. In October the gain was largest for plants just below the typical price.",
        "model": "model",
        "baseline": "same hour yesterday",
        "error_chart": "Price forecast error by day",
    },
    "tr": {
        "page_title": "Santral yarın çalışsın mı?",
        "title": "Santral yarın çalışsın mı?",
        "intro": "Bir santral operatörü için (1 MW, örneğin bir gaz ünitesi). Saat saat hangi teklifin kabul "
                 "edileceğine ihale karar verir; ama teklif edilecek blok, bildirilecek gaz ve vardiya fiyatlar "
                 "açıklanmadan, 12:30'dan önce sabitlenmek zorundadır. Fiyatlar: Türkiye gün öncesi piyasası (PTF), "
                 "USD/MWh. Tahminler her pazartesi yeniden eğitilen bir modelden; hiçbir model seçiminde kullanılmamış "
                 "günler için ({first} – {last}). Veri: EPİAŞ Şeffaflık Platformu; hava verisi "
                 "[Open-Meteo.com](https://open-meteo.com/).",
        "language": "Dil",
        "your_plant": "Santraliniz",
        "cost": "Değişken maliyet (yakıt), USD/MWh",
        "min_hours": "En az çalışma süresi, saat",
        "start_cost": "Bir açılışın maliyeti, USD",
        "assumptions": "Varsayımlar, gerçek bir santralin rakamları değil. Santral günde en fazla bir kesintisiz "
                       "blokta çalışır ve fiyatı etkilemez.",
        "tab_plan": "Yarının planı",
        "tab_trust": "Güvenebilir miyim?",
        "delivery_day": "Teslim günü",
        "forecast_made": "Tahmin **{day}** günü, 12:30 teklif saatinden önce yapıldı.",
        "show_actual": "Gerçekte ne olduğunu göster",
        "forecast": "tahmin",
        "actual": "gerçek",
        "your_cost": "maliyetiniz {cost}",
        "hour": "saat",
        "plan_chart": "Saatlik fiyat tahmini (yeşil: teklif edilecek blok, gri: maliyetin üstünde ama blok dışında, "
                      "kırmızı: maliyetin altında)",
        "hover_hour": "%{{x}}:00 — %{{y:.1f}} USD/MWh",
        "run": "**Hazırlan: {block} bloğunu teklif et.** Açılış maliyeti sonrası beklenen kâr: MW başına yaklaşık "
               "**{profit} USD**.",
        "stay_off": "**Yarın açma.** {cost} USD/MWh yakıt maliyetinde, en az {hours} saatlik hiçbir bloğun {start} "
                    "USD açılış maliyetini karşılaması beklenmiyor.",
        "block": "{first:02d}:00–{end:02d}:00 ({hours} saat)",
        "off": "kapalı",
        "outcome_title": "Kararın sonucu (gerçek fiyatlarla kâr, MW başına USD)",
        "follow_model": "Modeli izlemek",
        "follow_baseline": "Dünün aynı saatini izlemek",
        "perfect_best": "En iyisi (geleceği bilmek)",
        "runs": "çalışma: {block}",
        "day_error": "Bu gün tahminin ortalama hatası: {error} USD/MWh.",
        "trust_intro": "Demonun {days} gününün tamamı: her tahmin bir gün önce yapıldı, sonra gerçek fiyatlarla "
                       "karşılaştırıldı. Çıta: *dünün aynı saati*, herkesin bedavaya sahip olduğu tahmin.",
        "model_mae": "Model hatası (MAE, USD/MWh)",
        "baseline_mae": "Dünün aynı saati (MAE)",
        "days_better": "Modelin daha iyi olduğu gün",
        "of": "{wins} / {days} gün",
        "pays_off": "Santralinize kazandırıyor mu? ({days} gün, MW başına USD)",
        "always_run": "Hep çalış",
        "perfect": "Geleceği bilmek",
        "gain_earned": "{cost} USD/MWh yakıt maliyetinde, dünün fiyatları yerine modeli izlemek {days} günde "
                       "**MW başına {amount} USD** fazla kazandırdı.",
        "gain_lost": "{cost} USD/MWh yakıt maliyetinde, dünün fiyatları yerine modeli izlemek {days} günde "
                     "**MW başına {amount} USD** kaybettirdi.",
        "gain_chart": "Modelin dünün aynı saatine göre ek kârı, santral maliyetine göre (MW başına USD)",
        "hover_cost": "maliyet %{{x}}: %{{y:,.0f}} USD",
        "cost_axis": "değişken maliyet, USD/MWh",
        "gain_note": "Tahmin en az uçlarda işe yarar: çok ucuz bir santral zaten hep çalışır, çok pahalı bir santral "
                     "hiç çalışmaz. Ekim'de en büyük kazanç, tipik fiyatın biraz altındaki santrallerde oldu.",
        "model": "model",
        "baseline": "dünün aynı saati",
        "error_chart": "Günlük fiyat tahmini hatası",
    },
}

DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "tr": ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "tr": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim",
                 "Kasım", "Aralık"]}


def text(lang: str, key: str, **values) -> str:
    return TEXTS[lang][key].format(**values)


def long_date(day: date, lang: str, year=True) -> str:
    """'Sunday 11 October 2026' / '11 Ekim 2026 Pazar' (year optional)."""
    weekday, month = DAYS[lang][day.weekday()], MONTHS[lang][day.month - 1]
    tail = f" {day.year}" if year else ""
    return f"{weekday} {day.day:02d} {month}{tail}" if lang == "en" else f"{day.day} {month}{tail} {weekday}"


def short_date(day: date, lang: str, year=False) -> str:
    """'11 Oct' / '11 Eki' (year optional)."""
    month = MONTHS[lang][day.month - 1][:3]
    return f"{day.day:02d} {month}" + (f" {day.year}" if year else "")


def number(value: float, lang: str, decimals=0) -> str:
    """1,308.5 in English, 1.308,5 in Turkish."""
    formatted = f"{value:,.{decimals}f}"
    return formatted if lang == "en" else formatted.replace(",", "_").replace(".", ",").replace("_", ".")
