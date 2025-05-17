# Detekcija Prijevara Kreditnim Karticama - Mašinsko Učenje

## Opis projekta

Ovaj projekat implementira sistem za detekciju fraudulentnih transakcija kreditnim karticama koristeći različite algoritme mašinskog učenja. Sistem koristi anonimizirani skup podataka sa transakcijama evropskih korisnika kreditnih kartica.

## Tehnologije

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js
- **Backend**: Python, Flask
- **Mašinsko učenje**: Scikit-learn, XGBoost, imbalanced-learn
- **Ostalo**: Pandas, NumPy, Joblib

## Instalacija i pokretanje

1. Klonirajte repozitorijum:
   ```bash
   git clone https://github.com/belilovi02/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Instalirajte potrebne pakete:
   ```bash
   pip install -r requirements.txt
   ```

3. Pokrenite obradu podataka i treniranje modela:
   ```bash
   python data_processing.py
   ```

4. Pokrenite Flask server:
   ```bash
   python app.py
   ```

5. Otvorite `index.html` u web pregledniku

## Struktura projekta

```
credit-card-fraud-detection/
├── static/           # Statički fajlovi (CSS, JS)
├── templates/        # HTML šabloni
├── models/           # Sačuvani modeli i skaleri
├── data/             # Skup podataka (nije uključen u repozitorijum)
├── data_processing.py# Skripta za obradu podataka
├── app.py            # Flask aplikacija
└── requirements.txt  # Lista Python zavisnosti
```

## Korišteni algoritmi

1. **Random Forest**
   - Preciznost: 99.96%
   - Odziv: 85.71%
   - F1-score: 87.27%

2. **XGBoost**
   - Preciznost: 99.85%
   - Odziv: 87.76%
   - F1-score: 66.67%

3. **Logistička regresija**
   - Preciznost: 97.43%
   - Odziv: 91.84%
   - F1-score: 10.94%

## Kako koristiti aplikaciju

1. Unesite iznos transakcije
2. Popunite PCA karakteristike (V1-V28)
3. Odaberite model za analizu
4. Kliknite na "Provjeri transakciju"
5. Sistem će prikazati da li je transakcija legitimna ili sumnjiva

## Autori

## Licenca

