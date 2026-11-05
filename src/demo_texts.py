"""Texts of the demo app (app/streamlit_app.py) in English and Turkish, with date and number formatting per language.

Every text is looked up by key with `text(lang, key, **values)`; both languages must have the same keys
(tests/test_demo_texts.py). Placeholders use str.format: {days}, {hours}, ...
"""

from datetime import date

LANGUAGES = {"en": "English", "tr": "Türkçe"}

TEXTS = {
    "en": {
        "page_title": "Tomorrow's electricity price",
        "title": "Tomorrow's electricity price",
        "intro": "Turkish day-ahead market (PTF), USD/MWh, read on the day before, before the 12:30 offer deadline. "
                 "Forecasts by a model retrained every Monday, for days no model choice used ({first} – {last}). "
                 "Data: EPİAŞ Transparency Platform; weather data by [Open-Meteo.com](https://open-meteo.com/).",
        "your_job": "Your flexible consumption",
        "n_hours": "How many hours the job must run per day",
        "job_note": "A flexible job is electricity use whose timing you can choose, for example a water pump, a cold store "
                    "or a mill that must run 4 hours a day at 1 MW but not at fixed hours. The hours need not be "
                    "consecutive. The app puts them in the cheapest hours of the forecast and charges the actual prices "
                    "of those hours. The site is assumed too small to move the price.",
        "tab_price": "Tomorrow's price",
        "tab_value": "What the cheap hours are worth",
        "tab_trust": "Can I trust it?",
        "delivery_day": "Delivery day",
        "forecast_made": "Forecast made on **{day}**, before the 12:30 offer deadline.",
        "bulletin": "**{day}:** average **{mean} USD/MWh**, evening peak (17–22) {peak}. Cheapest {n} hours: "
                    "**{cheap}**. Priciest 4 hours: {pricey}. Day type: **{type}**.",
        "deep midday dip": "deep midday dip",
        "midday dip": "midday dip",
        "flat day": "flat day",
        "m_mean": "Daily mean",
        "m_day": "Day 06–17",
        "m_peak": "Peak 17–22",
        "m_night": "Night 22–06",
        "m_spread": "Highest − lowest hour",
        "price_chart": "Forecast per hour (green: your {n} cheapest hours, red: the 4 priciest) with the band where "
                       "the price usually falls",
        "forecast": "forecast",
        "bars_cheap": "forecast: your {n} cheapest hours (where the job runs)",
        "bars_pricey": "forecast: the 4 priciest hours",
        "bars_other": "forecast: other hours",
        "band": "band (80% of past hours)",
        "actual": "actual",
        "hour": "hour",
        "hover_hour": "%{{x}}:00 — %{{y:.1f}} USD/MWh",
        "show_actual": "Show what actually happened",
        "outcome_title": "Your {n} hours at the actual prices (USD per MW)",
        "src_model": "Following the model",
        "src_baseline": "Following same hour yesterday",
        "src_perfect": "Knowing the prices",
        "src_none": "Not shifting (day's average)",
        "hours_used": "hours: {hours}",
        "day_check": "This day: average error {error} USD/MWh · the actual price was inside the band in {inside} of "
                     "24 hours · day type forecast: {forecast_type}, actual: {actual_type}.",
        "changes_title": "What is different from today (daily means, known the day before)",
        "inp_load": "Load forecast (hourly average)",
        "inp_wind": "Wind forecast (hourly average)",
        "inp_temp": "Temperature forecast (5-city average)",
        "changes_note": "Large number: tomorrow's daily average. Small number: the change from today (minus = lower than "
                        "today). Load and wind are EPİAŞ's own forecasts for tomorrow. Shown side by side only: the model "
                        "uses many inputs together, so a change here is not the cause of the price.",
        "value_intro": "Your flexible job runs {n} hours a day at 1 MW. Each forecast puts it in the {n} hours it expects "
                       "to be cheapest; the job then pays the actual prices of those hours. Shown: the average price "
                       "paid per MWh. Lower is better.",
        "value_test": "October 2026 ({days} days, never seen by any model choice)",
        "value_validation": "Validation, 7 quarters 2025–2026 ({days} days)",
        "avg_paid": "average price paid, USD/MWh",
        "captured": "Share of the possible saving captured: model **{model}**, same hour yesterday **{baseline}** "
                    "(100% = knowing the prices).",
        "day_count": "Day by day: the model was cheaper on **{model} days**, yesterday on {baseline}, equal on {equal}.",
        "per_year": "Kept up for a year, that is about **{amount} USD per MW** more saving than following yesterday "
                    "(the measured daily average × 365).",
        "captured_chart": "Share of the possible saving captured, by hours per day (validation, {days} days)",
        "hover_captured": "%{{x}} hours: %{{y:.0f}}%",
        "hours_axis": "hours per day",
        "trust_intro": "All {days} demo days, each forecast made on the day before and compared with the actual "
                       "prices afterwards. Reference: *same hour yesterday*, a forecast anyone has for free.",
        "model_mae": "Model error (MAE, USD/MWh)",
        "baseline_mae": "Same hour yesterday (MAE)",
        "days_better": "Days the model was better",
        "of": "{wins} of {days}",
        "coverage": "Actual price inside the band",
        "coverage_note": "aim: 80%",
        "hits": "Cheapest {n} hours found",
        "hits_note": "yesterday: {baseline}",
        "types": "Day type right",
        "types_note": "yesterday: {baseline}",
        "by_hour_chart": "Model error by hour of day (MAE, USD/MWh): midday, when solar can push the price to zero, "
                         "is the hardest",
        "model": "model",
        "baseline": "same hour yesterday",
    },
    "tr": {
        "page_title": "Yarının elektrik fiyatı",
        "title": "Yarının elektrik fiyatı",
        "intro": "Türkiye gün öncesi piyasası (PTF), USD/MWh; bir gün önce, 12:30 teklif saatinden önce okunur. "
                 "Tahminler her pazartesi yeniden eğitilen bir modelden; hiçbir model seçiminde kullanılmamış günler "
                 "için ({first} – {last}). Veri: EPİAŞ Şeffaflık Platformu; hava verisi "
                 "[Open-Meteo.com](https://open-meteo.com/).",
        "your_job": "Esnek tüketiminiz",
        "n_hours": "İşin günde kaç saat çalışması gerekiyor",
        "job_note": "Esnek iş, saatini sizin seçebildiğiniz elektrik tüketimidir: örneğin günde 4 saat, 1 MW güçle "
                    "çalışması gereken ama hangi saatte çalışacağı serbest olan bir su pompası, soğuk hava deposu ya "
                    "da değirmen. Saatlerin art arda olması gerekmez. Uygulama işi tahminin en ucuz saatlerine koyar "
                    "ve o saatlerin gerçek fiyatını öder. Tesisin fiyatı etkilemeyecek kadar küçük olduğu varsayılır.",
        "tab_price": "Yarının fiyatı",
        "tab_value": "Ucuz saatlerin değeri",
        "tab_trust": "Güvenebilir miyim?",
        "delivery_day": "Teslim günü",
        "forecast_made": "Tahmin **{day}** günü, 12:30 teklif saatinden önce yapıldı.",
        "bulletin": "**{day}:** ortalama **{mean} USD/MWh**, akşam puantı (17–22) {peak}. En ucuz {n} saat: "
                    "**{cheap}**. En pahalı 4 saat: {pricey}. Gün tipi: **{type}**.",
        "deep midday dip": "derin öğlen çukuru",
        "midday dip": "öğlen düşüşü",
        "flat day": "düz gün",
        "m_mean": "Günlük ortalama",
        "m_day": "Gündüz 06–17",
        "m_peak": "Puant 17–22",
        "m_night": "Gece 22–06",
        "m_spread": "En yüksek − en düşük saat",
        "price_chart": "Saatlik tahmin (yeşil: en ucuz {n} saatiniz, kırmızı: en pahalı 4 saat) ve fiyatın genelde "
                       "düştüğü bant",
        "forecast": "tahmin",
        "bars_cheap": "tahmin: en ucuz {n} saatiniz (işin çalışacağı saatler)",
        "bars_pricey": "tahmin: en pahalı 4 saat",
        "bars_other": "tahmin: diğer saatler",
        "band": "bant (geçmiş saatlerin %80'i)",
        "actual": "gerçek",
        "hour": "saat",
        "hover_hour": "%{{x}}:00 — %{{y:.1f}} USD/MWh",
        "show_actual": "Gerçekte ne olduğunu göster",
        "outcome_title": "{n} saatiniz gerçek fiyatlarla (MW başına USD)",
        "src_model": "Modeli izlemek",
        "src_baseline": "Dünün aynı saatini izlemek",
        "src_perfect": "Fiyatları bilmek",
        "src_none": "Kaydırmamak (günün ortalaması)",
        "hours_used": "saatler: {hours}",
        "day_check": "Bu gün: ortalama hata {error} USD/MWh · bandın içinde kalan saat: {inside} / 24 · "
                     "gün tipi tahmini: {forecast_type}, gerçek: {actual_type}.",
        "changes_title": "Bugünden farkı (günlük ortalamalar, bir gün önceden bilinir)",
        "inp_load": "Yük tahmini (saatlik ortalama)",
        "inp_wind": "Rüzgâr tahmini (saatlik ortalama)",
        "inp_temp": "Sıcaklık tahmini (5 şehir ortalaması)",
        "changes_note": "Büyük sayı: yarının günlük ortalaması. Küçük sayı: bugüne göre fark (eksi = yarın bugünden düşük). "
                        "Yük ve rüzgâr EPİAŞ'ın yarın için kendi tahminleri. Yalnızca yan yana gösterim: model birçok "
                        "girdiyi birlikte kullanır, buradaki bir değişim fiyatın nedeni sayılmaz.",
        "value_intro": "Esnek işiniz günde {n} saat, 1 MW güçle çalışıyor. Her tahmin işi en ucuz bulduğu {n} saate "
                       "koyuyor; iş o saatlerin gerçek fiyatını ödüyor. Gösterilen: MWh başına ödenen ortalama fiyat. "
                       "Düşük olan iyidir.",
        "value_test": "Ekim 2026 ({days} gün, hiçbir model seçiminde görülmedi)",
        "value_validation": "Validation, 2025–2026'nın 7 çeyreği ({days} gün)",
        "avg_paid": "ödenen ortalama fiyat, USD/MWh",
        "captured": "Mümkün tasarrufun yakalanan payı: model **{model}**, dünün aynı saati **{baseline}** "
                    "(%100 = fiyatları bilmek).",
        "day_count": "Gün gün: model **{model} gün** daha ucuz, dünün aynı saati {baseline} gün, {equal} gün eşit.",
        "per_year": "Bir yıl sürse dünü izlemeye göre MW başına yaklaşık **{amount} USD** daha fazla tasarruf "
                    "(ölçülen günlük ortalama × 365).",
        "captured_chart": "Mümkün tasarrufun yakalanan payı, günde gereken saate göre (validation, {days} gün)",
        "hover_captured": "%{{x}} saat: yüzde %{{y:.0f}}",
        "hours_axis": "günde saat",
        "trust_intro": "Demonun {days} gününün tamamı: her tahmin bir gün önce yapıldı, sonra gerçek fiyatlarla "
                       "karşılaştırıldı. Çıta: *dünün aynı saati*, herkesin bedavaya sahip olduğu tahmin.",
        "model_mae": "Model hatası (MAE, USD/MWh)",
        "baseline_mae": "Dünün aynı saati (MAE)",
        "days_better": "Modelin daha iyi olduğu gün",
        "of": "{wins} / {days} gün",
        "coverage": "Gerçek fiyat bandın içinde",
        "coverage_note": "hedef: %80",
        "hits": "En ucuz {n} saati bulma",
        "hits_note": "dünün aynı saati: {baseline}",
        "types": "Gün tipi doğru",
        "types_note": "dünün aynı saati: {baseline}",
        "by_hour_chart": "Saate göre model hatası (MAE, USD/MWh): güneşin fiyatı sıfıra çekebildiği öğlen saatleri "
                         "en zoru",
        "model": "model",
        "baseline": "dünün aynı saati",
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


def percent(share: float, lang: str, decimals=0) -> str:
    """0.78 -> '78%' in English, '%78' in Turkish."""
    value = number(100 * share, lang, decimals=decimals)
    return f"{value}%" if lang == "en" else f"%{value}"


def hour_text(runs: list[tuple[int, int]]) -> str:
    """[(10, 14), (18, 19)] -> '10:00–14:00, 18:00–19:00'."""
    return ", ".join(f"{first:02d}:00–{end:02d}:00" for first, end in runs)
