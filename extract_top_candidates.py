import sys

def extract_top_candidates(input_file, n=3):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    top_candidates = [[] for _ in range(n)]
    for i in range(0, len(lines), n+1):  # +1 for the blank line separator
        for j in range(n):
            if i+j < len(lines):
                top_candidates[j].append(lines[i+j].strip())

    # Save to separate files
    for idx, candidates in enumerate(top_candidates):
        output_file = input_file.replace('.txt', f'_top{idx+1}.txt')
        with open(output_file, 'w') as file:
            file.write('\n'.join(candidates))
            file.write('\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_top_candidates.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    extract_top_candidates(input_file)

