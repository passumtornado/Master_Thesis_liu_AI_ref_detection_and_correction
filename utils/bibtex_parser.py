from pathlib import Path
import os
import bibtexparser
import json
import requests



def bibtex_parser(file_path: str):
    """Parse a BibTeX file and return a list of entries in standardized dict format"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BibTeX file not found: {file_path}")
        library = bibtexparser.parse_file(file_path)
        return library
    except Exception as e:
        print(f"Error parsing BibTeX file {file_path}: {e}")
        return None

def process_bibtexfile(file_path: str, output_json: str = "references.json"):
    """Load BibTeX entries from a file"""
    library = bibtex_parser(file_path)
    if library is None:
        return None
    output_path = export_library_to_json(library, output_json)
    if output_path is None:
        print("Failed to export library to JSON.")
        return None 
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Processed {len(data)} entries from {file_path}")
    return data

def download_bibtexfile(url: str, download_dir: str = './downloads') -> str:
    """Download a BibTeX file from a URL and save it to the specified directory
    
    Args:
        url: URL to fetch BibTeX from
        download_dir: Directory to save the file (relative to bibtex/bibtex_files/)
    
    Returns:
        Path to the downloaded file
    """
    base_dir = Path("bibtex/bibtex_files")
    os.makedirs(base_dir, exist_ok=True)
    save_dir = base_dir / download_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract filename from URL
    filename = Path(url.split("?")[0]).name
    if not filename.endswith(".bib"):
        filename = f"downloaded_{Path(url).stem}.bib"
    
    local_path = save_dir / filename
    
    try:
        # Download the BibTeX file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        
        # Save the file
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Downloaded BibTeX file to {local_path}")
        return str(local_path)
    except Exception as e:
        print(f"Error downloading BibTeX file from {url}: {e}")
        return None    

# function to export the library entries into a JSON file
def export_library_to_json(library, filename: str = 'references.json'):
    """Export BibTeX library entries to a JSON file"""
    test_dir = Path("bibtex/bibtex_files")
    os.makedirs(test_dir, exist_ok=True)
    out_path = test_dir / filename
    try:
        serializable = []
        for entry in library.entries:
            entry_dict = {
                "key": entry.key,                        # cite key  e.g. "smith2020"
                "entry_type": entry.entry_type,                       # e.g. "article"
                "fields":     {f.key: f.value for f in entry.fields}  # all fields as plain dict
            }
            serializable.append(entry_dict)

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2, ensure_ascii=False)
        return out_path

    except Exception as e:
        print(f"Error exporting library to JSON: {e}")
        return None


if __name__ == "__main__":
    data = process_bibtexfile("bibtex/bibtex_files/references.bib","processed_references.json")
    print(data)

