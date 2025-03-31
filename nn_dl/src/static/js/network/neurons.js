function drawNeuronValues() {
    // Draw values for input neurons
    svg.selectAll(".inputNeuron")
        .each(function (d, i) {
            const neuron = d3.select(this);
            const value = inputNeuronValues[i];
            drawValueForNeuron(neuron, value, "inputNeuronValue");
        });

    // Draw values for hidden neurons
    svg.selectAll(".hiddenNeuron")
        .each(function (d, i) {
            const neuron = d3.select(this);
            const value = hiddenNeuronValues[i];
            drawValueForNeuron(neuron, value, "hiddenNeuronValue");
        });

    // Draw values for output neurons
    svg.selectAll(".outputNeuron")
        .each(function (d, i) {
            const neuron = d3.select(this);
            const value = outputNeuronValues[i];
            drawValueForNeuron(neuron, value, "outputNeuronValue");
        });
}

function drawValueForNeuron(neuron, value, className) {
    const x = parseFloat(neuron.attr("cx"));
    const y = parseFloat(neuron.attr("cy"));
    svg.append("text")
        .attr("class", className)
        .attr("x", x)
        .attr("y", y)
        .attr("dy", "0.35em")  // to vertically center the text
        .attr("text-anchor", "middle")
        .text(value.toFixed(2))
        .attr("font-size", "10px")
        .attr("fill", "black")
}


function drawOutputNeuronsLabels() {
    const textOffset = 30;  // Adjust this value as needed

    svg.selectAll(".neuron")
        .filter(function () {
            return d3.select(this).attr("cx") == (noHiddenLayers + 2) * layerGap;
        })
        .each(function (d, i) {
            const neuron = d3.select(this);
            const x = parseFloat(neuron.attr("cx")) + textOffset;
            const y = parseFloat(neuron.attr("cy"));

            svg.append("text")
                .attr("x", x)
                .attr("y", y)
                .attr("dy", "0.35em")  // to vertically center the text
                .attr("text-anchor", "start")
                .text(i)  // The number corresponding to the neuron
                .attr("font-size", "14px")
                .attr("fill", "black")
        });
}

const transitionDuration = 2000; // duration of the transition animation

function highlightHiddenNeurons() {
    // Change the neuron color
    const selectedNeuron = svg.selectAll(".hiddenNeuron")
        .filter((d, i) => hiddenNeuronValues[i] >= 0.5);
    selectedNeuron.transition().duration(transitionDuration).style("fill", "#c73a52");
    // Filter and highlight hidden neurons with values >= 0.5
    svg.selectAll(".hiddenNeuronValue")
        .filter((d, i) => hiddenNeuronValues[i] >= 0.5)
        .transition().duration(transitionDuration).style("fill", "white");
}

function highlightOutputNeuron() {
    // Find the max output neuron value and its index
    const maxOutputValue = Math.max(...outputNeuronValues);
    const maxOutputValueIndex = outputNeuronValues.indexOf(maxOutputValue);
    // Change the neuron color
    const selectedNeuron = svg.selectAll(".outputNeuron")
        .filter((d, i) => i === maxOutputValueIndex);
    selectedNeuron.transition().duration(transitionDuration + 2000).style("fill", "#c73a52");
    // Change the text color of the neuron with the highest value
    const selectedNeuronText = svg.selectAll(".outputNeuronValue")
        .filter((d, i) => i === maxOutputValueIndex);
    selectedNeuronText.transition().duration(transitionDuration + 2000).style("fill", "white");
}
