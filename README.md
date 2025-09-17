# Product deduplication challenge

## Steps taken/reasoning

1. Load data into Postgresql for initial exploration. I didn’t put much effort into this and used both spark and pandas frames, as I faced some issues when trying to load the data with a spark dataframe. Exploration allowed me to tell that there are no  reliable ID columns, and see that there are no full duplicate rows.
2. My first attempt at finding duplicate products was to fuzzy match product titles. I chose titles, as they are more descriptive than the names. This worked for most cases, but proved problematic for cases where product differed by a couple of numbers in the name e.g. “1144 carbon steel” versus “1214 carbon steel”. They were getting a similarity score above the threshold of 90% that I was aiming to use. To try and solve this, I replaced numbers with their equivalent text version. So “1” became “one”, “2” became “two” etc. This got rid of some of the false matches, but I encountered the issue again for products with _very_ similar models: “Epic Smart Digital Door Lock ES-F700G” and “Epic Smart Digital Door Lock ES-7000K”. In this case, the difference was so small that even with the text-representation, a match of 96% was found. At this point, I considered an llm check for the grey area of 95<match score<100, but there were too many rows and running took forever, so I gave up on the approach.

3. Next up, I decided	to switch to cosine similarity to hopefully get better results. Using that, and setting the threshold limit to 92.5 provided the most satisfactory results.

4. Once I identified matching pairs by checking for cosine similarity > 92.5, the next step was to consolidate the rows into a single entry. To do this, I used GraphFrame to compute connected components.

5. Once each row had a component id, I grouped the data frame by component and used the max values for each row. I chose this approach as there is no timestamp to tell which the most recent values are, and it seemed like the most straightforward approach (for example prices are likely to go up, not down, populated values are prioritised over empty ones). This was an arbitrary decision, and in a real scenario this choice would have to be discussed on a column by column basis with relevant stakeholders.

6. The final deduplicated data is under data/products_deduped.snappy.parquet

## Other comments
I’d like to mention that the format of the data was maintained to resemble the initial one, but for a real life implementation I’d add more processing. For example, I might add a separate product_prices table to have separate rows for the different values (min, max, exact). Or simply separate the product price into more columns (price_min, price_max, price_amount, price_currency). For this challenge I felt that doing that for all columns wouldn’t have brought much value compared to the time it would have taken to implement, as I don’t think cleaning and restructuring data was the focus.
