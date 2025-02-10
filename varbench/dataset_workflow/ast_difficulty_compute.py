from pylatexenc.latexwalker import LatexEnvironmentNode, LatexWalker


original_tex = open("dataset/tikz/bee_eyes/input.tex").read()
solution_tex = open("dataset/tikz/bee_eyes/solutions/solution1.tex").read()

original_w = LatexWalker(original_tex)
solution_w = LatexWalker(solution_tex)

(nodelist, pos, len_) = original_w.get_latex_nodes(pos=0)
print(nodelist[10])
