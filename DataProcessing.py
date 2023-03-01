import camelot

abc = camelot.read_pdf("data/pdf_tables.pdf")
print(abc[0].df)