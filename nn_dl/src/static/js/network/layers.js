function drawInputNeurons() {
    // Draw Input Neurons
    if (noInputNeurons <= noMaxNeuronsDisplay) {
        for (let i = 0; i < noInputNeurons; i++) {
            circleGroup.append("circle")
                .attr("class", "neuron inputNeuron")
                .attr("cx", neuronRadius * 2)
                .attr("cy", 2 * neuronRadius + (i + 1) * neuronGap)  // Adjusted for padding
                .attr("r", neuronRadius);
        }
    } else {
        // Draw the first neurons
        for (let i = 0; i < ((noMaxNeuronsDisplay - 1) / 2); i++) {
            circleGroup.append("circle")
                .attr("class", "neuron inputNeuron")
                .attr("cx", neuronRadius * 2)
                .attr("cy", 2 * neuronRadius + (i + 1) * neuronGap)  // Adjusted for padding
                .attr("r", neuronRadius);
        }
        // Draw the ellipsis
        circleGroup.append("text")
            .attr("x", neuronRadius * 2)
            .attr("y", 2 * neuronRadius + (((noMaxNeuronsDisplay - 1) / 2) + 1) * neuronGap)  // Position it after the 8th neuron
            .attr("dy", "0.35em")  // to vertically center the text
            .attr("text-anchor", "middle")
            .text("...")
            .attr("font-size", "11px")
            .attr("fill", "black");
        // Draw the last neurons
        for (let i = 0; i < ((noMaxNeuronsDisplay - 1) / 2); i++) {
            circleGroup.append("circle")
                .attr("class", "neuron inputNeuron")
                .attr("cx", neuronRadius * 2)
                .attr("cy", 2 * neuronRadius + (i + 8) * neuronGap)  // Adjusted for padding and ellipsis
                .attr("r", neuronRadius);
        }
    }
}

function drawHiddenLayerNeurons(p) {
    let actualNeuronsDisplayed = Math.min(noNeuronsPerHiddenLayer, noMaxNeuronsDisplayHiddenLayers);
    let centerYAdjusted = actualNeuronsDisplayed * neuronGap / 2;

    if (noNeuronsPerHiddenLayer <= noMaxNeuronsDisplayHiddenLayers) {
        for (let m = 0; m < noNeuronsPerHiddenLayer; m++) {
            circleGroup.append("circle")
                .attr("class", "neuron hiddenNeuron")
                .attr("cx", (p + 2) * layerGap)
                .attr("cy", 2 * neuronRadius + centerY + (m - noNeuronsPerHiddenLayer / 2) * neuronGap)  // Adjusted for padding
                .attr("r", neuronRadius);
        }
    } else {
        // Draw the first neurons
        for (let m = 0; m < ((noMaxNeuronsDisplayHiddenLayers - 1) / 2); m++) {
            circleGroup.append("circle")
                .attr("class", "neuron hiddenNeuron")
                .attr("cx", (p + 2) * layerGap)
                .attr("cy", 2 * neuronRadius + centerYAdjusted + (m - (noMaxNeuronsDisplayHiddenLayers - 1) / 2) * neuronGap)  // Adjusted for padding
                .attr("r", neuronRadius);
        }
        // Draw the ellipsis
        circleGroup.append("text")
            .attr("x", (p + 2) * layerGap)
            .attr("y", 2 * neuronRadius + centerYAdjusted)  // Position it at the center
            .attr("dy", "0.35em")  // to vertically center the text
            .attr("text-anchor", "middle")
            .text("...")
            .attr("font-size", "11px")
            .attr("fill", "black");
        // Draw the last neurons
        for (let m = 0; m < ((noMaxNeuronsDisplayHiddenLayers - 1) / 2); m++) {
            circleGroup.append("circle")
                .attr("class", "neuron hiddenNeuron")
                .attr("cx", (p + 2) * layerGap)
                .attr("cy", 2 * neuronRadius + centerYAdjusted + (m + 1) * neuronGap)  // Adjusted for padding and ellipsis
                .attr("r", neuronRadius);
        }
    }
}

function drawOutputNeurons() {
    // Draw Output Neurons
    circleGroup.selectAll(".outputNeuron")
        .data(outputNeuronValues)
        .enter()
        .append("circle")
        .attr("class", "neuron outputNeuron")
        .attr("cx", (noHiddenLayers + 2) * layerGap)
        .attr("cy", (d, i) => 2 * neuronRadius + centerY + (i - noOutputNeurons / 2) * neuronGap)  // Adjusted for padding
        .attr("r", neuronRadius);
}


function drawLayers() {
    drawInputNeurons();
    for (let p = 0; p < noHiddenLayers; p++) {
        drawHiddenLayerNeurons(p);
    }
    drawOutputNeurons();
}
