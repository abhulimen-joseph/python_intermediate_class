# Scrape books from All products | Books to Scrape - Sandbox
from selenium import webdriver
from selenium.webdriver.common.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time 

driver = webdriver.Chrome(ChromeDriverManager().install())
# (Book name, price, stock status (in stock or out of stock), rating, description, product information, category (poetry, fiction, historical fiction, etc)
# Scrape the first 5 pages (20 books per page)

# print(f"Found {len(books)} books on the website")
def web_scrapper(driver):
    no_pages = 5

    books_list = []
    # for i in range(1, no_pages + 1):
    for i in range(1, no_pages + 1):
        driver.get(f"https://books.toscrape.com/catalogue/page-{i}.html")
        time.sleep(3)
        
        books = driver.find_elements(By.CSS_SELECTOR, "li.col-md-3")
        for i in range(len(books)):
            try:
                book_name = books[i].find_element(By.CSS_SELECTOR, "h3").text
                price = books[i].find_element(By.CSS_SELECTOR, "p.price_color").text
                stock_status = books[i].find_element(By.CSS_SELECTOR, "p.instock.availability").text
                description_url = books[i].find_element(By.TAG_NAME, "a")
                description_url.click()
                time.sleep(3)
                description = driver.find_element(By.XPATH, "//div[@id='product_description']/following::p[1]").text
                
                product_information_table = driver.find_element(By.CSS_SELECTOR, "table.table-striped")
                product_information = {}

                rows = product_information_table.find_elements(By.TAG_NAME, "tr")
                for row in rows:
                    th = row.find_element(By.TAG_NAME, "th")
                    td = row.find_element(By.TAG_NAME, "td")

                    key = th.text.strip()
                    value = td.text.strip()

                    product_information[key] = value

                driver.back()
                time.sleep(3)
                
                book =  {
                    "book name": book_name,
                    "Price": price,
                    "Stock_status": stock_status,
                    "Description": description,
                    "Product Information": product_information
                    
                }
                books_list.append(f"Book {i+1}:\n"
                        f"  Name: {book["book name"]}\n"
                        f"  Price: {book["Price"]}\n"
                        f"  Stock status: {book["Stock_status"]}\n"
                        f" Description: {book["Description"]}\n"
                        f" Product Information: {book["Product Information"]}\n")
                
            except:
                print(f"{i+1}. [Could not find details]")

        for book_entry in books_list:
            print(book_entry, end="")

# Scrape 10-20 distinct quote authors from Quotes to Scrape
# (Name, nationality, description, date of birth)

def quote_scrapper(driver):
    driver.get("https://quotes.toscrape.com/")
    quotes_element = driver.find_elements(By.CSS_SELECTOR, "quote")
    author_details = []
    while len(author_details) < 20:
        if len(author_details) >=20:
            break
        for i, quote_ele in enumerate(quotes_element):
            try:
                name = quote_ele.find_element(By.CSS_SELECTOR, "author" ).text
                about_url = quote_ele.find_element(By.XPATH, "//a[text() = '(about)']" ).get_attribute("href")
                about_url.get()
                time.sleep(3)

                location = driver.get(By.CSS_SELECTOR, "author-born-location").text
                if len(location):
                    nationality = location.split(",")[-1].strip()

                description = driver.find_element(By.CSS_SELECTOR, "author_description").text[:200]
                dob = driver.find_element(By.CSS_SELECTOR, "author-born-date").text
                
            
                author_details.append(f"Author {i+1}:\n"
                                    f"Name: {name}\n"
                                    f"Nationality: {nationality}\n"
                                    f"Description: {description}\n"
                                    f"D.O.B: {dob}\n")
                driver.back()
                time.sleep(3)
            except:
                print(f"{i+1}: [Couldn't find element]")
            
        if len(author_details) < 20:
            button = driver.find_element(By.XPATH, "//a[contains(text()= 'Next'])").get_attribute("href")
            button.click()
            time.sleep(3)
                    
            

    for author in author_details:
        print(author, end="")
        
def run2():
    try:
        authors = quote_scrapper(driver)
    finally:
        driver.quit()


# Build a scraper that will scrape a random page from Wikipedia

def wikipedia_scraper(driver):
    """Scrape random Wikipedia pages"""
    driver.get("https://en.wikipedia.org/wiki/Special:Random")
    page_details = []
    
    while len(page_details) < 5:  # Get 5 random pages
        try:
            # Get page title
            title = driver.find_element(By.CSS_SELECTOR, "#firstHeading").text
            print(f"\n📖 Page {len(page_details)+1}: {title}")
            
            # Get page URL
            page_url = driver.current_url
            
            # Get summary (first paragraph)
            try:
                summary = driver.find_element(
                    By.CSS_SELECTOR, "#mw-content-text .mw-parser-output > p"
                ).text
                if len(summary) > 300:
                    summary = summary[:300] + "..."
            except:
                summary = "No summary available"
            
            # Get first image
            try:
                image = driver.find_element(
                    By.CSS_SELECTOR, ".mw-parser-output .infobox img, .mw-parser-output .thumb img"
                )
                image_url = image.get_attribute("src")
            except:
                image_url = "No image"
            
            # Get categories
            try:
                categories_elements = driver.find_elements(
                    By.CSS_SELECTOR, "#mw-normal-catlinks ul li a"
                )
                categories = [cat.text for cat in categories_elements[:3]]  # First 3 only
            except:
                categories = []
            
            # Get infobox data
            infobox_data = {}
            try:
                infobox = driver.find_element(By.CSS_SELECTOR, "table.infobox")
                rows = infobox.find_elements(By.TAG_NAME, "tr")
                
                for row in rows[:5]:  # First 5 rows only
                    try:
                        th = row.find_element(By.TAG_NAME, "th")
                        td = row.find_element(By.TAG_NAME, "td")
                        key = th.text.strip()
                        value = td.text.strip()
                        if key and value:
                            infobox_data[key] = value
                    except:
                        continue
            except:
                infobox_data = {}
            
            # Add to results
            page_details.append(f"Page {len(page_details)+1}:\n"
                              f"Title: {title}\n"
                              f"URL: {page_url}\n"
                              f"Summary: {summary}\n"
                              f"Image: {image_url}\n"
                              f"Categories: {', '.join(categories) if categories else 'None'}\n")
            
            # Add infobox data if exists
            if infobox_data:
                page_details[-1] += "Key Facts:\n"
                for key, value in infobox_data.items():
                    page_details[-1] += f"  • {key}: {value}\n"
            
            page_details[-1] += "-"*50 + "\n"
            
            time.sleep(2)
            
            # Go to another random page
            if len(page_details) < 5:
                driver.get("https://en.wikipedia.org/wiki/Special:Random")
                time.sleep(2)
                
        except Exception as e:
            print(f"Error: {e}")
            # Try another random page
            driver.get("https://en.wikipedia.org/wiki/Special:Random")
            time.sleep(2)
            continue
    
    return page_details



def run3():
    details = wikipedia_scraper()
    print(details)
    

  





