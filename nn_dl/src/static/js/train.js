document.addEventListener("DOMContentLoaded", function () {
    const N = 14; // or whatever size you want for NxN grid
    const M = 10; // for 0-9 in Mx1 grid

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

    // Generate output grid
    const outputGrid = document.getElementById('outputGrid');
    for (let i = 0; i < M; i++) {
        const cell = document.createElement('div');
        cell.classList.add('grid-cell');
        cell.textContent = i;
        cell.addEventListener('click', function () {
            // Remove toggled class from all other cells
            Array.from(outputGrid.children).forEach(c => c.classList.remove('toggled'));
            cell.classList.add('toggled');
        });
        outputGrid.appendChild(cell);
    }

    // Set the first cell as toggled by default
    outputGrid.firstChild.classList.add('toggled');

    // Submit button logic
    document.getElementById('submitBtn').addEventListener('click', function () {
        const inputValues = Array.from(inputGrid.children).map(cell => cell.classList.contains('toggled') ? 1 : 0);
        const outputValue = Array.from(outputGrid.children).findIndex(cell => cell.classList.contains('toggled'));

        if (inputValues.includes(1)) {
            console.log('Input:', inputValues);
            console.log('Expected Output:', outputValue);

            // Send data to Spring backend
            fetch('/train/submit', {
                method: 'POST', headers: {
                    'Content-Type': 'application/json'
                }, body: JSON.stringify({
                    input: inputValues, expectedOutput: outputValue
                })
            }).then(response => response.json()).then(data => {
                console.log(data);
            });
        } else {
            alert('Please toggle at least one cell in the input image.');
        }

        // Reload the page after submitting
        location.reload();
    });
});

