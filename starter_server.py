
import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    app = FirecrawlApp(api_key=api_key)
    
    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    metadata_file = os.path.join(path, "scraped_metadata.json")
    
    existing_metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                existing_metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {metadata_file}, starting fresh.")

    scraped_providers = []

    for provider, url in websites.items():
        logger.info(f"Scraping {provider} at {url}")
        
        try:
            scrape_result = app.scrape(url, formats=formats)
            
            # Prepare metadata entry
            timestamp = datetime.now().isoformat()
            domain = urlparse(url).netloc
            
            content_files = {}
            for fmt in formats:
                content = getattr(scrape_result, fmt, "")
                if content:
                    filename = f"{provider}_{fmt}.txt"
                    file_path = os.path.join(path, filename)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    content_files[fmt] = filename
            
            # Handle metadata safely
            title = "Unknown Title"
            description = "No description"
            if hasattr(scrape_result, 'metadata'):
                title = getattr(scrape_result.metadata, 'title', "Unknown Title")
                description = getattr(scrape_result.metadata, 'description', "No description")

            metadata_entry = {
                "provider_name": provider,
                "url": url,
                "domain": domain,
                "scraped_at": timestamp,
                "formats": formats,
                "success": "true",
                "content_files": content_files,
                "title": title,
                "description": description
            }
            
            existing_metadata[provider] = metadata_entry
            scraped_providers.append(provider)
            logger.info(f"Successfully scraped {provider}")
            
        except Exception as e:
            logger.error(f"Failed to scrape {provider}: {e}")
            # Optionally record failure in metadata
            existing_metadata[provider] = {
                "provider_name": provider,
                "url": url,
                "scraped_at": datetime.now().isoformat(),
                "success": "false",
                "error": str(e)
            }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=4)

    return scraped_providers

@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    
    if not os.path.exists(SCRAPE_DIR):
        return json.dumps({"error": "No scraped content found."})

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    if not os.path.exists(metadata_file):
        return json.dumps({"error": "Metadata file not found."})

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid metadata file."})

    # Search for the identifier
    found_data = None
    
    # Check if identifier matches a provider key directly
    if identifier in metadata:
        found_data = metadata[identifier]
    else:
        # Search by url or domain
        for provider, data in metadata.items():
            if identifier == data.get("url") or identifier == data.get("domain"):
                found_data = data
                break
    
    if found_data:
        # Read content files
        content_data = {}
        if "content_files" in found_data:
            for fmt, filename in found_data["content_files"].items():
                file_path = os.path.join(SCRAPE_DIR, filename)
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content_data[fmt] = f.read()
        
        found_data["content"] = content_data
        return json.dumps(found_data, indent=2)
    else:
        return json.dumps({"error": f"No information found for {identifier}"})

if __name__ == "__main__":
    mcp.run(transport="stdio")