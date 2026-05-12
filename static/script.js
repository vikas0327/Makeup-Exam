document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const predictBtn = document.getElementById('predict-btn');
    const predictionsContainer = document.getElementById('predictions-container');
    const statusMessage = document.getElementById('status-message');
    const spinner = document.getElementById('loading-spinner');
    
    // Graph Error Handling
    const graphImg = document.getElementById('graph-img');
    const graphError = document.getElementById('graph-error');
    
    graphImg.addEventListener('error', () => {
        graphImg.classList.add('hidden');
        graphError.classList.remove('hidden');
    });

    // Execute prediction on button click
    predictBtn.addEventListener('click', () => {
        handlePredictionRequest();
    });

    // Execute prediction on Enter key
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handlePredictionRequest();
        }
    });

    function handlePredictionRequest() {
        const text = textInput.value.trim();
        if (text.length > 0) {
            hideStatus();
            spinner.classList.remove('hidden');
            predictionsContainer.innerHTML = '';
            fetchPredictions(text);
        } else {
            showStatus("Please enter a sentence fragment first.");
            renderEmptyState();
        }
    }

    async function fetchPredictions(text) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });
            
            spinner.classList.add('hidden');
            const data = await response.json();
            
            if (response.status === 503) {
                showStatus(data.error + " - You need to run 'python train.py' in the terminal first!");
                renderErrorState();
                return;
            }
            
            if (response.status !== 200) {
                showStatus("Error: " + (data.error || "Unknown error occurred"));
                renderErrorState();
                return;
            }
            
            if (data.predictions && data.predictions.length > 0) {
                renderPredictions(data.predictions);
            } else {
                renderNoPredictions();
            }
        } catch (error) {
            spinner.classList.add('hidden');
            console.error("Error fetching predictions:", error);
            showStatus("Connection error. Is the Flask server running?");
            renderErrorState();
        }
    }

    function renderPredictions(predictions) {
        predictionsContainer.innerHTML = '';
        predictions.forEach((pred, index) => {
            const card = document.createElement('div');
            card.className = 'prediction-card';
            card.style.animation = `fadeInUp 0.4s ease-out ${index * 0.1}s both`;
            
            const confidencePercent = (pred.confidence * 100).toFixed(1);
            
            card.innerHTML = `
                <span class="word">${pred.word}</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: ${confidencePercent}%"></div>
                </div>
                <span class="confidence-text">${confidencePercent}% Confidence</span>
            `;
            
            card.addEventListener('click', () => {
                const currentText = textInput.value;
                const space = currentText.endsWith(' ') ? '' : ' ';
                textInput.value = currentText + space + pred.word + ' ';
                textInput.focus();
                
                // Immediately trigger next prediction
                handlePredictionRequest();
            });
            
            predictionsContainer.appendChild(card);
        });
    }

    function showStatus(message) {
        statusMessage.querySelector('span').textContent = message;
        statusMessage.classList.remove('hidden');
    }

    function hideStatus() {
        statusMessage.classList.add('hidden');
    }

    function renderEmptyState() {
        predictionsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-keyboard"></i>
                <p>Click "Predict" to see the next words</p>
            </div>
        `;
    }

    function renderNoPredictions() {
        predictionsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-ghost"></i>
                <p>No predictions available for this context.</p>
            </div>
        `;
    }
    
    function renderErrorState() {
        predictionsContainer.innerHTML = `
            <div class="empty-state" style="color: #ef4444;">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <p>Model Unavailable</p>
            </div>
        `;
    }
});
