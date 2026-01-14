# Scrape books from All products | Books to Scrape - Sandbox
from selenium import webdriver
from selenium.webdriver.common.by import By
import time 

driver = webdriver.Chrome()
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

# Build a scraper that will scrape a random page from Wikipedia

