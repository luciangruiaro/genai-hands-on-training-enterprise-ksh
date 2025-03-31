document.addEventListener("DOMContentLoaded", function () {
    const N = 14; // or whatever size you want for NxN grid

// Dynamically set the grid-template-columns property
    document.getElementById('inputGrid').style.gridTemplateColumns = `repeat(${N}, 1fr)`;


    // Generate input grid
    const inputGrid = document.getElementById('inputGrid');
    for (let i = 0; i < N * N; i++) {
        const cell = document.createElement('div');
        cell.classList.add('grid-cell');
        cell.addEventListener('click', function () {
            cell.classList.toggle('toggled');
        });
        inputGrid.appendChild(cell);
    }

    // Submit button logic
    document.getElementById('submitBtn').addEventListener('click', function () {
        const inputValues = Array.from(inputGrid.children).map(cell => cell.classList.contains('toggled') ? 1 : 0);

        if (inputValues.includes(1)) {
            console.log('Input:', inputValues);

            // Send data to Spring backend
            fetch('test/submit', {
                method: 'POST', headers: {
                    'Content-Type': 'application/json'
                }, body: JSON.stringify({
                    input: inputValues
                })
            }).then(response => response.json()).then(data => {
                console.log(data);
            });
        } else {
            alert('Please toggle at least one cell in the input image.');
        }

        // Reload the page after submitting
        // location.reload();
    });
});

