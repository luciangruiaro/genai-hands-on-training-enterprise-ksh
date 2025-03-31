def test_sn(new_input, s_neuron, test_context=""):
    print(
        f"[{test_context}] Prediction for {new_input} => {s_neuron.classify(new_input)} [{s_neuron.output(new_input)}]")
    s_neuron.print_state()
