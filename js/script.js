document.addEventListener('DOMContentLoaded', function() {
    // Generiši polja za PCA karakteristike
    const pcaContainer = document.getElementById('pcaFeaturesContainer');
    for (let i = 1; i <= 28; i++) {
        const col = document.createElement('div');
        col.className = 'col-md-4 col-6';
        col.innerHTML = `
            <div class="mb-2">
                <label for="v${i}" class="form-label small">V${i}</label>
                <input type="number" class="form-control form-control-sm" id="v${i}" step="0.000001">
            </div>
        `;
        pcaContainer.appendChild(col);
    }

    // Generisanje testnih podataka
    document.getElementById('testDataBtn').addEventListener('click', function() {
        generateTestData();
        
        // Vizuelna povratna informacija
        this.textContent = 'Novi testni podaci generisani!';
        this.classList.remove('btn-outline-secondary');
        this.classList.add('btn-success');
        
        setTimeout(() => {
            this.textContent = 'Generiši testne podatke';
            this.classList.remove('btn-success');
            this.classList.add('btn-outline-secondary');
        }, 1500);
    });

    // Funkcija za generisanje testnih podataka
    function generateTestData() {
        const scenarios = [
            {type: 'legitimate', amountRange: [10, 500], valueRange: [-1, 1]},
            {type: 'fraud', amountRange: [500, 2000], valueRange: [-5, 5]},
            {type: 'borderline', amountRange: [100, 300], valueRange: [-2, 2]}
        ];
        
        const scenario = scenarios[Math.floor(Math.random() * scenarios.length)];
        const form = document.getElementById('transactionForm');
        
        // Resetujemo klasu forme
        form.className = '';
        setTimeout(() => {
            form.classList.add(scenario.type);
        }, 10);
        
        // Generisanje iznosa
        const amount = (Math.random() * 
                      (scenario.amountRange[1] - scenario.amountRange[0]) + 
                      scenario.amountRange[0]).toFixed(2);
        document.getElementById('amount').value = amount;
        
        // Generisanje PCA karakteristika
        for (let i = 1; i <= 28; i++) {
            const val = Math.random() > 0.8 ? 
                  (Math.random() * 
                   (scenario.valueRange[1] - scenario.valueRange[0]) + 
                   scenario.valueRange[0]) :
                  (Math.random() * 2 - 1);
            
            document.getElementById(`v${i}`).value = val.toFixed(6);
        }
        
        // Nasumično odabran model
        const models = ['random_forest', 'xgboost', 'logistic_regression'];
        const randomModel = models[Math.floor(Math.random() * models.length)];
        document.getElementById('modelType').value = randomModel;
    }

    // Učitaj metrike modela
    fetchModelMetrics();

    // Obrada forme
    document.getElementById('transactionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const submitBtn = this.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Procesiram...';
        
        const amount = parseFloat(document.getElementById('amount').value);
        const modelType = document.getElementById('modelType').value;
        
        // Prikupi sve PCA karakteristike
        const pcaFeatures = {};
        for (let i = 1; i <= 28; i++) {
            pcaFeatures[`v${i}`] = parseFloat(document.getElementById(`v${i}`).value);
        }
        
        // Pripremi podatke za slanje
        const formData = {
            amount: amount,
            model_type: modelType,
            time: Date.now() / 1000,
            ...pcaFeatures
        };

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                showResult(result.is_fraud, result.probability, amount, modelType);
            } else {
                alert('Došlo je do greške: ' + result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Došlo je do greške prilikom komunikacije sa serverom');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Provjeri transakciju';
        }
    });

    // Funkcija za prikaz rezultata
    function showResult(isFraud, probability, amount, modelType) {
        const resultCard = document.getElementById('resultCard');
        const resultHeader = document.getElementById('resultHeader');
        const resultContent = document.getElementById('resultContent');
        
        resultCard.style.display = 'block';
        
        if (isFraud) {
            resultCard.className = 'card shadow mt-4 fraud';
            resultHeader.className = 'card-header fraud';
            resultContent.innerHTML = `
                <div class="alert alert-danger">
                    <h4 class="alert-heading">⚠️ Upozorenje: Potencijalna prevara!</h4>
                    <p>Transakcija od ${amount.toFixed(2)} KM je označena kao sumnjiva.</p>
                    <hr>
                    <p class="mb-0">Vjerovatnoća prevare: ${(probability * 100).toFixed(2)}%</p>
                </div>
            `;
        } else {
            resultCard.className = 'card shadow mt-4 legitimate';
            resultHeader.className = 'card-header legitimate';
            resultContent.innerHTML = `
                <div class="alert alert-success">
                    <h4 class="alert-heading">✓ Transakcija je legitimna</h4>
                    <p>Transakcija od ${amount.toFixed(2)} KM je prošla provjeru.</p>
                    <hr>
                    <p class="mb-0">Vjerovatnoća prevare: ${(probability * 100).toFixed(2)}%</p>
                </div>
            `;
        }
        
        // Prikaži statistike modela
        displayModelStats(modelType);
    }

    // Funkcija za prikaz statistika modela
    function displayModelStats(modelType) {
        const modelStats = {
            'random_forest': {
                accuracy: 0.9996,
                precision: 0.8889,
                recall: 0.8571,
                f1: 0.8727
            },
            'xgboost': {
                accuracy: 0.9985,
                precision: 0.5375,
                recall: 0.8776,
                f1: 0.6667
            },
            'logistic_regression': {
                accuracy: 0.9743,
                precision: 0.0581,
                recall: 0.9184,
                f1: 0.1094
            }
        };
        
        const stats = modelStats[modelType];
        const modelStatsElement = document.getElementById('modelStats');
        
        if (stats) {
            modelStatsElement.innerHTML = `
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Tačnost (Accuracy)
                    <span class="badge bg-primary rounded-pill">${(stats.accuracy * 100).toFixed(2)}%</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Preciznost (Precision)
                    <span class="badge bg-primary rounded-pill">${(stats.precision * 100).toFixed(2)}%</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Odziv (Recall)
                    <span class="badge bg-primary rounded-pill">${(stats.recall * 100).toFixed(2)}%</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    F1-score
                    <span class="badge bg-primary rounded-pill">${(stats.f1 * 100).toFixed(2)}%</span>
                </li>
            `;
        }
    }

    // Funkcija za učitavanje metrika modela
    async function fetchModelMetrics() {
        try {
            const response = await fetch('http://localhost:5000/model_metrics');
            const data = await response.json();
            
            if (data.success) {
                // Postavi izveštaje o klasifikaciji
                data.metrics.forEach(metric => {
                    if (metric.Model === 'Random Forest') {
                        document.getElementById('rfClassificationReport').textContent = formatClassificationReport(metric);
                    } else if (metric.Model === 'XGBoost') {
                        document.getElementById('xgbClassificationReport').textContent = formatClassificationReport(metric);
                    } else if (metric.Model === 'Logistic Regression') {
                        document.getElementById('lrClassificationReport').textContent = formatClassificationReport(metric);
                    }
                });
            }
        } catch (error) {
            console.error('Error fetching model metrics:', error);
        }
    }

    // Pomoćna funkcija za formatiranje izveštaja
    function formatClassificationReport(metrics) {
        return `
              precision    recall  f1-score   support

           0     ${metrics['Precision (Class 0)'].toFixed(4)}    ${metrics['Recall (Class 0)'].toFixed(4)}    ${metrics['F1 (Class 0)'].toFixed(4)}     56864
           1     ${metrics['Precision (Class 1)'].toFixed(4)}    ${metrics['Recall (Class 1)'].toFixed(4)}    ${metrics['F1 (Class 1)'].toFixed(4)}        98

    accuracy                          ${metrics.Accuracy.toFixed(4)}     56962
   macro avg      ${((metrics['Precision (Class 0)'] + metrics['Precision (Class 1)'])/2).toFixed(4)}    ${((metrics['Recall (Class 0)'] + metrics['Recall (Class 1)'])/2).toFixed(4)}    ${((metrics['F1 (Class 0)'] + metrics['F1 (Class 1)'])/2).toFixed(4)}     56962
weighted avg      ${metrics.Accuracy.toFixed(4)}    ${metrics.Accuracy.toFixed(4)}    ${metrics.Accuracy.toFixed(4)}     56962
        `;
    }

    generateTestData();
});