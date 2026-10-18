import re

def parse_bibtex(input_text):
    # Split the text into individual BibTeX entries
    entries = input_text.strip().split('@')
    
    # Initialize an empty list to store the dictionaries
    bib_list = []
    # Iterate through each entry
    for entry in entries:
        if not entry:
            continue  # Skip any empty entries
        
        # Extract the citation key
        key_match = re.search(r'\{(.+?),', entry)
        if key_match:
            key = key_match.group(1)
        else:
            raise ValueError('Key not found')  # Raise an error if key is not found
            continue  # Skip if key is not found
        
        # Extract the title
        title_match = re.search(r'title=\{(.+?)\}', entry, re.DOTALL)
        if title_match:
            title = title_match.group(1)
        else:
            print(f'Title not found for key: {key}')  # Print a warning if title is not found
            continue  # Skip if title is not found
        
        # Append the dictionary to the list
        bib_list.append({'key': key, 'title': title})
    breakpoint()
    return bib_list

def check_for_duplicate_titles(bib_list):
    titles_seen = set()
    duplicates = []

    for entry in bib_list:
        title = entry['title']
        
        # Remove everything after colon for comparison
        title_trimmed = title.split(':')[1].strip() if ':' in title else title.strip()
        # Check if title or trimmed title is already seen
        if title in titles_seen or title_trimmed in titles_seen:
            duplicates.append(entry)
        else:
            titles_seen.add(title)
            titles_seen.add(title_trimmed)
    
    return duplicates

# Example usage
txt_file = './bibtex.txt'
with open(txt_file, 'r') as file:
    bibtex_text = file.read()

# Parse the BibTeX entries
parsed_bibtex = parse_bibtex(bibtex_text)

# Check for duplicates
duplicates = check_for_duplicate_titles(parsed_bibtex)

# Output the duplicates
if duplicates:
    print("Duplicate entries found:")
    for entry in duplicates:
        print(entry)
else:
    print("No duplicates found.")