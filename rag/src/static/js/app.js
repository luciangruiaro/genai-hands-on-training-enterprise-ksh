function callGet() {
    fetch("/hello")
        .then(response => response.json())
        .then(data => {
            document.getElementById("responseBox").textContent = JSON.stringify(data, null, 2);
        })
        .catch(error => {
            document.getElementById("responseBox").textContent = "Error: " + error;
        });
}

function callPost() {
    const name = document.getElementById("nameInput").value;
    const age = parseInt(document.getElementById("ageInput").value);

    const payload = {name, age};

    fetch("/hello", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("responseBox").textContent = JSON.stringify(data, null, 2);
        })
        .catch(error => {
            document.getElementById("responseBox").textContent = "Error: " + error;
        });
}
