from RecBole.recbole.quick_start import run_recbole
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context # for downloading dataset

dataset = [
    "book-crossing",
    "lastfm",
    "ml-1m",
    "Amazon_Luxury_Beauty",
    "Amazon_Digital_Music",
    "Amazon_Industrial_and_Scientific",
]

models = [
    "Pop",
    ]



for data in dataset:
    for model in models:      
        curr_result = run_recbole(
                                model=model,
                                dataset=data,
                                )
