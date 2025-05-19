def json_to_latex_table(json_data, caption=None, label=None):
    """
    Convert JSON data (list of dicts) to a LaTeX table string.

    Parameters:
        json_data (list): List of dictionaries representing table rows.
        caption (str): Optional caption for the table.
        label (str): Optional label for referencing the table.

    Returns:
        str: LaTeX table code.
    """
    if not json_data:
        raise ValueError("Empty JSON data")

    columns = list(json_data[0].keys())

    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{" + " | ".join(["l"] * len(columns)) + "}")
    latex.append("\\hline")

    header = " & ".join(columns) + " \\\\"
    latex.append(header)
    latex.append("\\hline")

    for row in json_data:
        row_str = " & ".join(str(row[col]) for col in columns) + " \\\\"
        latex.append(row_str)

    latex.append("\\hline")
    latex.append("\\end{tabular}")

    if caption:
        latex.append(f"\\caption{{{caption}}}")
    if label:
        latex.append(f"\\label{{{label}}}")

    latex.append("\\end{table}")

    return "\n".join(latex)

def parse_percentage(value):
    """Convert percentage string like '80%' to float."""
    return int(value.strip('%'))

def convert_to_json(data_str):
    """Convert multiline tabular string to list of dicts in desired JSON format."""
    json_list = []
    lines = data_str.strip().splitlines()
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        parts = line.split()
        
        if len(parts) < 5:
            continue  # Incomplete row, skip
        
        total_clients = int(parts[0])
        rounds = int(parts[1])
        clients_per_round_percent = parse_percentage(parts[2])
        accuracy = float(parts[3])
        byzantine_clients = int(parts[4])

        json_list.append({
            "total_clients": total_clients,
            "rounds": rounds,
            "clients_per_round_percent": clients_per_round_percent,
            "byzantine_clients": byzantine_clients,
            "accuracy": accuracy
        })
    
    return json_list


mnist_data_rtime = """
4             1                     100%                     0.9511               0
4             1                     80%                      0.9505               0
4             1                     70%                      0.9510               0
5             1                     100%                     0.9474               0
5             1                     80%                      0.9463               0
5             1                     70%                      0.9479               0
6             1                     100%                     0.9425               0
6             1                     80%                      0.9405               0
6             1                     70%                      0.9421               0
4             2                     100%                     0.9539               0
4             2                     80%                      0.9515               0
4             2                     70%                      0.9527               0
5             2                     100%                     0.9495               0
5             2                     80%                      0.9484               0
5             2                     70%                      0.9493               0
6             2                     100%                     0.9452               0
6             2                     80%                      0.9439               0
6             2                     70%                      0.9427               0
4             1                     100%                     0.9154               1
4             1                     100%                     0.7508               2
4             1                     100%                     0.7492               3
4             1                      80%                     0.8630               1
4             1                      80%                     0.7489               2
4             1                      80%                     0.7466               3
4             1                      70%                     0.7511               1
4             1                      70%                     0.7511               2
4             1                      70%                     0.7511               3
5             1                     100%                     0.9277               1
5             1                     100%                     0.7883               2
5             1                     100%                     0.7462               3
5             1                      80%                     0.9049               1
5             1                      80%                     0.7506               2
5             1                      80%                     0.7506               3
5             1                      70%                     0.9455               1
5             1                      70%                     0.8440               2
5             1                      70%                     0.8572               3
6             1                     100%                     0.9251               1
6             1                     100%                     0.8570               2
6             1                     100%                     0.7419               3
6             1                      80%                     0.8997               1
6             1                      80%                     0.8942               2
6             1                      80%                     0.7410               3
6             1                      70%                     0.9411               1
6             1                      70%                     0.9140               2
6             1                      70%                     0.7395               3
4             2                     100%                     0.9533               1
4             2                     100%                     0.9524               2
4             2                     100%                     0.9523               3
4             2                      80%                     0.9528               1
4             2                      80%                     0.9514               2
4             2                      80%                     0.9521               3
4             2                      70%                     0.9528               1
4             2                      70%                     0.9517               2
4             2                      70%                     0.9517               3
5             2                     100%                     0.9511               1
5             2                     100%                     0.9496               2
5             2                     100%                     0.9504               3
5             2                      80%                     0.9496               1
5             2                      80%                     0.9483               2
5             2                      80%                     0.9481               3
5             2                      70%                     0.9480               1
5             2                      70%                     0.9488               2
5             2                      70%                     0.9474               3
6             2                     100%                     0.9446               1
6             2                     100%                     0.9439               2
6             2                     100%                     0.9433               3
6             2                      80%                     0.9428               1
6             2                      80%                     0.9437               2
6             2                      80%                     0.9420               3
6             2                      70%                     0.9428               1
6             2                      70%                     0.9434               2
6             2                      70%                     0.9432               3
"""

mnist_data = """
5                 5               100%                    0.9758               0
5                 5               80%                     0.9762               0
5                 5               70%                     0.9788               0
7                 5               100%                    0.9733               0
7                 5               80%                     0.9737               0
7                 5               70%                     0.9727               0
10                5               100%                    0.9677               0
10                5               80%                     0.9682               0
10                5               70%                     0.9681               0
5                 5              100%                     0.9736               1
5                 5              100%                     0.9691               2
5                 5              100%                     0.9321               3
5                 5               80%                     0.9738               1
5                 5               80%                     0.9729               2
5                 5               80%                     0.9579               3
5                 5               70%                     0.9759               1
5                 5               70%                     0.9750               2
5                 5               70%                     0.9658               3
7                 5              100%                     0.9718               1
7                 5              100%                     0.9678               2
7                 5              100%                     0.9589               3
7                 5               80%                     0.9689               1
7                 5               80%                     0.9620               2
7                 5               80%                     0.9608               3
7                 5               70%                     0.9688               1
7                 5               70%                     0.9700               2
7                 5               70%                     0.9701               3
10                5              100%                     0.9676               1
10                5              100%                     0.9656               2
10                5              100%                     0.9625               3
10                5               80%                     0.9657               1
10                5               80%                     0.9638               2
10                5               80%                     0.9610               3
10                5               70%                     0.9662               1
10                5               70%                     0.9615               2
10                5               70%                     0.9519               3
"""

fashionmnist_data = """
5          5            100%                0.914               0
5          5            80%                 0.913               0
5          5            70%                 0.9132              0
7          5            100%                0.9117              0
7          5            80%                 0.91                0
7          5            70%                 0.9118              0
10         5            100%                0.9101              0
10         5            80%                 0.9067              0
10         5            70%                 0.9078              0
5          5            100%                0.9104              1
5          5            80%                 0.9092              1
5          5            70%                 0.9084              1
7          5            100%                0.9108              1
7          5            80%                 0.9093              1
7          5            70%                 0.9058              1
10         5            100%                0.9115              1
10         5            80%                 0.9072              1
10         5            70%                 0.9094              1
5          5            100%                0.8976              2
5          5            80%                 0.908               2
5          5            70%                 0.9096              2
7          5            100%                0.9042              2
7          5            80%                 0.9039              2
7          5            70%                 0.8959              2
10         5            100%                0.9062              2
10         5            80%                 0.9045              2
10         5            70%                 0.9028              2
5          5            100%                0.7859              3
5          5            80%                 0.8727              3
5          5            70%                 0.9011              3
7          5            100%                0.8955              3
7          5            80%                 0.9018              3
7          5            70%                 0.9049              3
10         5            100%                0.9063              3
10         5            80%                 0.8932              3
10         5            70%                 0.8986              3
"""

fashionmnist_data_rtime = """
4            1            100%           0.9027         0
4            1            80%            0.9006         0
4            1            70%            0.8908         0
4            2            100%           0.9012         0
4            2            80%            0.8997         0
4            2            70%            0.8960         0
5            1            100%           0.9005         0
5            1            80%            0.8987         0
5            1            70%            0.8968         0
5            2            100%           0.8978         0
5            2            80%            0.8974         0
5            2            70%            0.8938         0
6            1            100%           0.8976         0
6            1            80%            0.8968         0
6            1            70%            0.8953         0
6            2            100%           0.8940         0
6            2            80%            0.8922         0
6            2            70%            0.8938         0
4            1            100%           0.8982         1
4            1            80%            0.8600         1
4            1            70%            0.7664         1
4            2            100%           0.9023         1
4            2            80%            0.8966         1
4            2            70%            0.8975         1
5            1            100%           0.8987         1
5            1            80%            0.8844         1
5            1            70%            0.8978         1
5            2            100%           0.9010         1
5            2            80%            0.8982         1
5            2            70%            0.8972         1
6            1            100%           0.8949         1
6            1            80%            0.8793         1
6            1            70%            0.8790         1
6            2            100%           0.8958         1
6            2            80%            0.8932         1
6            2            70%            0.8981         1
4            1            100%           0.7323         2
4            1            80%            0.7121         2
4            1            70%            0.7071         2
4            2            100%           0.9028         2
4            2            80%            0.8963         2
4            2            70%            0.8982         2
5            1            100%           0.8057         2
5            1            80%            0.7584         2
5            1            70%            0.8621         2
5            2            100%           0.8990         2
5            2            80%            0.9002         2
5            2            70%            0.8962         2
6            1            100%           0.8666         2
6            1            80%            0.8790         2
6            1            70%            0.8913         2
6            2            100%           0.8923         2
6            2            80%            0.8953         2
6            2            70%            0.8940         2
4            1            100%           0.7110         3
4            1            80%            0.7096         3
4            1            70%            0.7090         3
4            2            100%           0.9027         3
4            2            80%            0.8974         3
4            2            70%            0.8948         3
5            1            100%           0.7099         3
5            1            80%            0.7113         3
5            1            70%            0.8298         3
5            2            100%           0.8981         3
5            2            80%            0.8996         3
5            2            70%            0.8957         3
6            1            100%           0.7117         3
6            1            80%            0.7222         3
6            1            70%            0.7069         3
6            2            100%           0.8922         3
6            2            80%            0.8909         3
6            2            70%            0.8881         3
"""
mnist_data = convert_to_json(mnist_data)
mnist_data_rtime = convert_to_json(mnist_data_rtime)
fashionmnist_data = convert_to_json(fashionmnist_data)
fashionmnist_data_rtime = convert_to_json(fashionmnist_data_rtime)

latex_code = json_to_latex_table(fashionmnist_data_rtime, caption="Model Hyperparameters", label="tab:model_hp")
print(latex_code)