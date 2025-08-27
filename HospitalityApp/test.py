import requests
import re
from collections import defaultdict

def decode_secret_message(url):
    """
    Decodes a secret message from a Google Doc by parsing Unicode characters
    and their coordinates, then displaying them in a 2D grid.
    
    Args:
        url (str): URL to the Google Doc containing the encoded message
        
    Returns:
        str: The decoded secret message
    """
    
    # Fetch the document content
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
    except requests.RequestException as e:
        print(f"Error fetching document: {e}")
        return ""
    
    # Extract Unicode characters and coordinates using regex
    # Look for patterns like: character at coordinates (x, y)
    pattern = r'([A-Z])\s*(?:at|@)\s*\(?(\d+)\s*,\s*(\d+)\)?'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    if not matches:
        # Try alternative pattern for different document formats
        pattern = r'(\d+)\s*,\s*(\d+)\s*:\s*([A-Z])'
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Reorder to match expected format (char, x, y)
            matches = [(match[2], match[0], match[1]) for match in matches]
    
    if not matches:
        print("No character coordinates found in the document")
        return ""
    
    # Create a dictionary to store characters at their coordinates
    grid = defaultdict(lambda: defaultdict(lambda: ' '))
    max_x, max_y = 0, 0
    
    for char, x_str, y_str in matches:
        x, y = int(x_str), int(y_str)
        grid[y][x] = char.upper()
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    
    # Create and display the grid
    print("Decoded grid:")
    result_lines = []
    
    for y in range(max_y + 1):
        line = ""
        for x in range(max_x + 1):
            line += grid[y][x]
        result_lines.append(line.rstrip())
        print(line.rstrip())
    
    # Extract the secret message by reading non-space characters
    secret_message = ""
    for line in result_lines:
        for char in line:
            if char != ' ':
                secret_message += char
    
    return secret_message

# Alternative version for when the document format is different
def decode_secret_message_alt(url):
    """
    Alternative decoder that tries different parsing approaches
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
    except requests.RequestException as e:
        print(f"Error fetching document: {e}")
        return ""
    
    # Multiple regex patterns to try
    patterns = [
        r'([A-Z])\s*(?:at|@)\s*\(?(\d+)\s*,\s*(\d+)\)?',
        r'(\d+)\s*,\s*(\d+)\s*:\s*([A-Z])',
        r'(\d+)\s+(\d+)\s+([A-Z])',
        r'([A-Z])\s+(\d+)\s+(\d+)'
    ]
    
    coordinates = []
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            if pattern == patterns[1] or pattern == patterns[2]:
                # Reorder to (char, x, y) format
                coordinates = [(match[2], int(match[0]), int(match[1])) for match in matches]
            else:
                coordinates = [(match[0].upper(), int(match[1]), int(match[2])) for match in matches]
            break
    
    if not coordinates:
        print("Could not parse coordinates from document")
        return ""
    
    # Build the grid
    grid = {}
    max_x = max_y = 0
    
    for char, x, y in coordinates:
        if (x, y) not in grid:
            grid[(x, y)] = char
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    # Display the grid
    print("Decoded message:")
    result = []
    
    for y in range(max_y + 1):
        row = []
        for x in range(max_x + 1):
            row.append(grid.get((x, y), ' '))
        line = ''.join(row).rstrip()
        result.append(line)
        print(line)
    
    # Extract message
    message = ''.join(char for line in result for char in line if char != ' ')
    return message

# Test with the provided URL
if __name__ == "__main__":
    test_url = "https://docs.google.com/document/d/e/2PACX-1vTER-wL5E8YC9pxDx43gk8eIds59GtUUk4nJo_ZWagbnrH0NFvMXIw6VWFLpf5tWTZIT9P9oLIoFJ6A/pub"
    
    print("Attempting to decode secret message...")
    message = decode_secret_message(test_url)
    
    if not message:
        print("Trying alternative parsing method...")
        message = decode_secret_message_alt(test_url)
    
    if message:
        print(f"\nSecret message: {message}")
    else:
        print("Could not decode the message. Please check the document format.")