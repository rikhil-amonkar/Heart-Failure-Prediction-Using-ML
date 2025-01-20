document.getElementById("input-information").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent form default submission behavior

    // Collect form data
    const data = {
        Age: document.getElementById("age").value,
        Sex: document.getElementById("sex").value,
        ChestPainType: document.getElementById("chest-pain").value,
        RestingBP: document.getElementById("resting-bp").value,
        Cholesterol: document.getElementById("cholesterol").value,
        FastingBS: document.getElementById("fasting-blood-sugar").value === "greater" ? 1 : 0,
        RestingECG: document.getElementById("ecg").value,
        MaxHR: document.getElementById("max-heart-rate").value,
        ExerciseAngina: document.getElementById("exercise-angina").value === "yes" ? 1 : 0,
        Oldpeak: document.getElementById("old-peak").value,
        ST_Slope: document.getElementById("st-slope").value
    };

    // Send data to the Flask endpoint
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    })
        .then(response => response.text())
        .then(result => {
            document.body.innerHTML = result; // Replace page content with result
        })
        .catch(error => console.error("Error:", error));
});
