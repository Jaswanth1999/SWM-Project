function analyzeReview() {
    var selectedModel = document.getElementById("modelDropdown").value;
    var userReview = document.getElementById("reviewInput").value;
	
    console.log(selectedModel)
    console.log(userReview)

    // Placeholder values for model metrics and accuracy
    var accuracy = "85%";
    var precision = "90%";
    var recall = "80%";
    var stats = {
	LSTM: {
	precision: "88%",
	recall: "90%",
	accuracy: "89%",
	},
	LR: {
	precision: "88%",
	recall: "90%",
	accuracy: "89%",
	},
	RF: {
	precision: "86%",
	recall: "84%",
	accuracy: "85%",
	},
	SVC: {
	precision: "88%",
	recall: "91%",
	accuracy: "89%",
	},
	BERT: {
	precision: "89%",
	recall: "89%",
	accuracy: "89%",
	},
	RBERT: {
	precision: "90%",
	recall: "90%",
	accuracy: "90%",
	}
}

    // Perform API call to your backend for sentiment analysis
    // Replace the following line with actual API call to your server
    // For example, using fetch() to send a POST request to your Flask or Django server
    fetch("http://127.0.0.1:5000/predict-sentiment", {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
	    "Access-Control-Allow-Origin": "*",
	    "model": selectedModel,
	    "text": userReview
        }
    })
    .then(response => response.json())
    .then(data => {
        // Display result and model metrics
        var resultElement = document.getElementById("result");
		resultElement.classList.toggle("d-none");
        resultElement.innerHTML = `Sentiment: ${data}<br>`;
        resultElement.innerHTML += `Accuracy: ${stats[selectedModel].accuracy}<br>`;
        resultElement.innerHTML += `Precision: ${stats[selectedModel].precision}<br>`;
        resultElement.innerHTML += `Recall: ${stats[selectedModel].recall}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}