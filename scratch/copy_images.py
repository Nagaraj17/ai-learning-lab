import shutil

src1 = r"C:\Users\Nagar\.gemini\antigravity-ide\brain\4eebfaca-d2b1-4283-bbb1-347fb35441c5\contextual_embedding_diagram_1785529978453.png"
dst1 = r"c:\Users\Nagar\source\repos\ai-learning-lab\projects\Week 3\contextual_embedding_diagram.png"

src2 = r"C:\Users\Nagar\.gemini\antigravity-ide\brain\4eebfaca-d2b1-4283-bbb1-347fb35441c5\attention_mechanism_diagram_1785529964711.png"
dst2 = r"c:\Users\Nagar\source\repos\ai-learning-lab\projects\Week 3\attention_mechanism_diagram.png"

shutil.copyfile(src1, dst1)
shutil.copyfile(src2, dst2)
print("SUCCESS")
