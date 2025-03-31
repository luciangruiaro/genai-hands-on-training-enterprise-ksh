// Function to draw lines between two layers
function connectLayers(layer1, layer2) {
    layer1.each(function () {
        const fromNeuron = d3.select(this);
        layer2.each(function () {
            const toNeuron = d3.select(this);
            lineGroup.append("line")
                .attr("class", "line")
                .attr("x1", fromNeuron.attr("cx"))
                .attr("y1", fromNeuron.attr("cy"))
                .attr("x2", toNeuron.attr("cx"))
                .attr("y2", toNeuron.attr("cy"));
        });
    });
}

function drawSynapses() {
    // Draw Lines
    const inputLayer = svg.selectAll(".neuron")
        .filter(function () {
            return d3.select(this).attr("cx") == neuronRadius * 2;
        });
    let previousLayer = inputLayer;
    for (let p = 0; p < noHiddenLayers; p++) {
        const currentHiddenLayer = svg.selectAll(".neuron")
            .filter(function () {
                return d3.select(this).attr("cx") == (p + 2) * layerGap;
            });
        connectLayers(previousLayer, currentHiddenLayer);
        previousLayer = currentHiddenLayer;
    }

    const outputLayer = svg.selectAll(".neuron")
        .filter(function () {
            return d3.select(this).attr("cx") == (noHiddenLayers + 2) * layerGap;
        });
    connectLayers(previousLayer, outputLayer);
}