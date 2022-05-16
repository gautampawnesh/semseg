import json, csv

VISTA_CONFIG_FILE = "vista_config.json"
# output file with color codes and className
VISTA_COLOR_CODE_FILE = "vistas_color_codes.csv"

with open(VISTA_CONFIG_FILE, 'r') as config_fo:
    data = json.load(config_fo)


header = ["Category", "ClassName", "ColorCode"]
with open(VISTA_COLOR_CODE_FILE, 'w', encoding='UTF8') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    for row_i in range(len(data["labels"])):
        row_data = data["labels"][row_i]
        writer.writerow([row_data["name"], row_data["readable"], ",".join(str(color) for color in row_data["color"])])
