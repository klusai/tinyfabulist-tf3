from transformers import PreTrainedTokenizerFast

# Load your trained tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="artifacts/ro_tokenizer_20250909122701.json"
)

# Your Romanian fable
text = """Bivolul posomorât și trecutul tocit

Soarele s-a ridicat deasupra savanei, pictând pajiștile în nuanțe calde de portocaliu. Un bivol morocănos, cunoscut pentru comportamentul său ursuz, călca apăsat prin iarba înaltă, tulburând liniștea dimineții. Celelalte animale îl priveau cu precauție, amintindu-și vechea ceartă dintre bivol și bătrânul elefant înțelept.

Vezi tu, cu mult timp în urmă, bivolul călcase din greșeală peste locul preferat de adăpat al elefantului, iar acesta fusese rănit. De atunci, bivolul fusese morocănos și răutăcios, iar elefantul devenise precaut în prezența lui.

Într-o zi, în timp ce bivolul păștea, s-a iscat o furtună puternică, iar bivolul s-a rătăcit în ceața deasă. S-a izbit de elefant, care se chinuia să-și găsească drumul spre casă. Bivolul, amintindu-și greșeala trecută, a simțit un fior de vinovăție.
     
         
„De ce mă ajuți?” a întrebat elefantul, surprins.

„Pentru că amândoi ne-am rătăcit și avem nevoie unul de altul”, a răspuns bivolul, fața lui ursuză înmuiindu-se.

În timp ce mergeau împreună, bivolul i-a povestit elefantului despre greșeala din trecut, iar elefantul și-a împărtășit durerea. Împreună, au lăsat vechea ceartă în urmă.

Din acea zi, bivolul și elefantul au devenit prieteni, iar savana s-a umplut din nou de râsete și pace. Bivolul a învățat că compasiunea ne poate ajuta să înțelegem trecutul celuilalt și că iertarea ne poate apropia."""

# Encode text
encoded = tokenizer(text)

# Show tokens instead of numbers
print("Tokens:")
print(encoded.tokens())
print("\nTotal tokens:", len(encoded["input_ids"]))
