function renderNeuron(data) {
    const inputValues = data.inputValues;
    const weights = data.weights;
    const biasWeight = data.biasWeight;
    const activated = data.activated;
    const output = data.output;
    const outputClassified = data.outputClassified;

    d3.select("#neuronVisual").html("");  // Clear previous SVG

    const svgWidth = 800;
    const svgHeight = 400;
    const neuronRadius = 60;
    const inputRadius = 20;
    const intermediaryRadius = 15;
    const outputRadius = 20;
    const neuronX = 3 * svgWidth / 4 + 40;
    const neuronY = svgHeight / 2;

    const svg = d3.select("#neuronVisual")
        .append("svg")
        .attr("width", svgWidth)
        .attr("height", svgHeight);

    const inputGap = 80;
    const inputStartY = neuronY - inputGap;
    const inputX = svgWidth / 6;

    const inputCircles = svg.selectAll(".input-circle")
        .data(inputValues)
        .enter()
        .append("circle")
        .attr("cx", inputX)
        .attr("cy", (d, i) => inputStartY + i * inputGap)
        .attr("r", inputRadius)
        .attr("class", "input-circle");

    inputCircles.each(function (d) {
        if (d === 1) {
            d3.select(this)
                .transition()
                .duration(2000)
                .style("fill", "#c73a52");
        }
    });

    svg.selectAll(".input-text")
        .data(inputValues)
        .enter()
        .append("text")
        .attr("x", inputX)
        .attr("y", (d, i) => inputStartY + i * inputGap)
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .text(d => d);

    const intermediaryX = neuronX - 2.5 * neuronRadius;
    const intermediaryY = neuronY;

    const aggregatorCircle = svg.append("circle")
        .attr("cx", intermediaryX)
        .attr("cy", intermediaryY)
        .attr("r", intermediaryRadius)
        .attr("class", "intermediary-circle");

    if (inputValues.some(val => val !== 0)) {
        aggregatorCircle.transition()
            .delay(2000)
            .duration(1000)
            .style("fill", "black");
    }

    for (let i = 0; i < inputValues.length; i++) {
        const inputY = inputStartY + i * inputGap;

        svg.append("line")
            .attr("x1", inputX + inputRadius)
            .attr("y1", inputY)
            .attr("x2", intermediaryX - intermediaryRadius)
            .attr("y2", inputY)
            .attr("class", "line")
            .transition()
            .duration(1000)
            .attr("y2", intermediaryY);

        svg.append("text")
            .attr("x", (inputX + intermediaryX) / 2)
            .attr("y", inputY - 10)
            .attr("text-anchor", "middle")
            .text(`W: ${weights[i].toFixed(2)}`);
    }

    svg.append("line")
        .attr("x1", intermediaryX + intermediaryRadius)
        .attr("y1", intermediaryY)
        .attr("x2", neuronX - neuronRadius)
        .attr("y2", neuronY)
        .attr("class", "line");

    svg.append("text")
        .attr("x", intermediaryX)
        .attr("y", neuronY + 30)
        .attr("text-anchor", "middle")
        .text(`Bias W: ${biasWeight.toFixed(2)}`);

    const neuronCircle = svg.append("circle")
        .attr("cx", neuronX)
        .attr("cy", neuronY)
        .attr("r", neuronRadius)
        .style("fill", "#0de7e7");
    const neuronOutputText = svg.append("text")
        .attr("fill", activated ? "black" : "white")
        .attr("x", neuronX)
        .attr("y", neuronY)
        .attr("dy", "-0.35em")
        .attr("text-anchor", "middle")
        .attr("opacity", 0)
        .text(`Output: ${output.toFixed(2)}`);

    neuronOutputText.transition()
        .delay(3000)
        .duration(3000)
        .attr("opacity", 1);

    if (activated) {
        neuronCircle.transition()
            .delay(3000)
            .duration(3000)
            .style("fill", "#c73a52");
    }

    const outputX = neuronX + neuronRadius + 50;
    svg.append("line")
        .attr("x1", neuronX + neuronRadius)
        .attr("y1", neuronY)
        .attr("x2", outputX)
        .attr("y2", neuronY)
        .attr("class", "line");

    const outputCircle = svg.append("circle")
        .attr("cx", outputX)
        .attr("cy", neuronY)
        .attr("r", outputRadius)
        .attr("class", "input-circle");

    const outputText = svg.append("text")
        .attr("x", outputX)
        .attr("y", neuronY)
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .attr("opacity", 0)
        .text(outputClassified);

    outputText.transition()
        .delay(7000)
        .duration(2000)
        .attr("opacity", 1);

    if (outputClassified === 1) {
        outputCircle.transition()
            .delay(7000)
            .duration(2000)
            .style("fill", "#c73a52");
    }

    if (data.epoch !== undefined) {
        document.getElementById("epoch-counter").textContent = data.epoch;
    }

}

document.addEventListener("DOMContentLoaded", function () {
    renderNeuron({
        inputValues,
        weights,
        biasWeight,
        activated,
        output,
        outputClassified
    });
});


document.getElementById("train-step-button").addEventListener("click", () => {
    fetch("/sn/train/step")
        .then(response => response.json())
        .then(data => {
            renderNeuron(data);
        });
});

document.getElementById("reset-button").addEventListener("click", () => {
    fetch("/sn/reset")
        .then(response => response.json())
        .then(data => {
            renderNeuron(data);
        });
});

document.getElementById("train-full-button").addEventListener("click", () => {
    fetch("/sn/train/full")
        .then(response => response.json())
        .then(data => {
            renderNeuron(data);
        });
});
