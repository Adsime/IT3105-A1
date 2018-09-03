import csv


def get_csv_cases(file):
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=';')
        cases = [[], []]
        for row in data:
            input = row[0:(len(row)-1)]
            for i, val in enumerate(input):
                input[i] = float(val)
            target = int(row[-1])
            cases[0].append(input)
            cases[1].append(target)
        return cases