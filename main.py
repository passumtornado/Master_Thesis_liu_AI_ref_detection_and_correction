# name = input("what is your name? ")
# file = open("name.txt","a")
# file.write(name + "\n")
# file.close()

# name = []

# with open("name.txt") as file:
#     for line in file:
#         name.append(line.rstrip())

# for name in sorted(name):
#     print('Hello ' + name + '!')    

#Simplfied
# with open("name.txt") as file:
#     for line in sorted(file):
#         print('Hello ' + line.rstrip() + '!')
 
        
# with open("student.csv") as file:
#     for line in file:
#         row = line.rstrip().split(",")
#         print(f"{row[0]} is in  {row[1]}")

#Unpacking
            # with open("student.csv") as file:
            #     for line in file:
            #         name, course = line.rstrip().split(",")
            #         print(f"{name} is in  {course}")




# from urllib import response

# import requests
# from bs4 import BeautifulSoup
# import re
# #Function to fetch Document
# def fetch_docs(doc_url):
#     try:
#        response = requests.get(doc_url)
#        response.raise_for_status()

#     except requests.RequestException as e:
#        print(f'document not found: {e}')
#        return
#     print(f'Document fetched successfully: {response.status_code}')
#     return response.text
    
    
# def decode_message(url):
#     html = fetch_docs(url) 
#     if not html:
#         print(f'Failed to fetch document from {url}')
#         return
    
#     # Parse HTML
#     soup = BeautifulSoup(html, "html.parser")

#     # Extract visible text
#     text = soup.get_text()
   

#     lines = text.splitlines()

#     # lines = html.text.strip().splitlines()
#     points = {}
#     min_X = min_Y = 0
#     max_X = max_Y = 0

#     # pattern = r'(\d+)([█░])(\d+)'

#     # matches = re.findall(pattern, text)
#     rows = soup.find_all("tr")
#     for row in rows:
#         cells = row.find_all("td")
#         if len(cells) >= 3:
#             x = cells[0].get_text(strip=True)
#             char = cells[1].get_text(strip=True)
#             y = cells[2].get_text(strip=True)
#             if x.isdigit() and y.isdigit():
#                 points[(int(x), int(y))] = char
#                 min_X = min(min_X, int(x))
#                 min_Y = min(min_Y, int(y))
#                 max_X = max(max_X, int(x))
#                 max_Y = max(max_Y, int(y))

#     # for match in matches:

#     #     x = int(match[0])
#     #     char = match[1]
#     #     y = int(match[2])

#     #     points[(x, y)] = char

#     #     min_X = min(min_X, x)
#     #     min_Y = min(min_Y, y)
#     #     max_X = max(max_X, x)
#     #     max_Y = max(max_Y, y)

#     if not points:
#         print(f'No valid points found')
#         return
#     output = []
#     for y in range(min_Y,max_Y+1):
#         row_chars = []
#         for x in range(min_X,max_X+1):
#             row_chars.append(points.get((x,y), ''))
#         output.append(''.join(row_chars))
        
        
#     final_output = '\n'.join(output)
#     with open("output.txt", "w") as file:
#         file.write(final_output.strip())
#     print(final_output)
            
        
# if __name__ == "__main__":
#     url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
#     #url=' https://docs.google.com/document/d/e/2PACX-1vTMOmshQe8YvaRXi6gEPKKlsC6UpFJSMAk4mQjLm_u1gmHdVVTaeh7nBNFBRlui0sTZ-snGwZM4DBCT/pub'
#     decode_message(url)
#     #fetch_docs(url)



import requests
from bs4 import BeautifulSoup


# Function to fetch document
def fetch_docs(doc_url):

    try:
        response = requests.get(doc_url)
        response.raise_for_status()

    except requests.RequestException as e:
        print(f"Document not found: {e}")
        return None

    print(f"Document fetched successfully: {response.status_code}")

    return response.text


# Function to decode and print secret message
def decode_message(url):

    html = fetch_docs(url)

    if not html:
        print(f"Failed to fetch document from {url}")
        return

    # Parse HTML
    soup = BeautifulSoup(html, "html.parser")

    # Store coordinates and characters
    points = {}

    max_x = 0
    max_y = 0

    # Extract table rows
    rows = soup.find_all("tr")

    for row in rows:

        cells = row.find_all("td")

        if len(cells) < 3:
            continue

        try:
            x = int(cells[0].get_text(strip=True))
            char = cells[1].get_text(strip=True)
            y = int(cells[2].get_text(strip=True))

        except ValueError:
            continue

        points[(x, y)] = char

        max_x = max(max_x, x)
        max_y = max(max_y, y)

    # Check if data exists
    if not points:
        print("No valid points found")
        return

    # Create grid filled with spaces
    grid = [[' '] * (max_y + 1) for _ in range(max_x + 1)]

    # Place characters into grid
    # Coordinates are flipped for correct orientation
    for (x, y), char in points.items():
        grid[x][y] = char

    # Convert grid into output text
    output = []

    for row in grid:
        output.append(''.join(row))

    final_output = '\n'.join(output)

    # Save output to file
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(final_output)

    # Print final decoded message
    print("\nDecoded Message:\n")
    print(final_output)


if __name__ == "__main__":

    url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"

    decode_message(url)