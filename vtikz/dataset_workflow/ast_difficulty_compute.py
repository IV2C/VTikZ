from TexSoup import TexSoup, TexNode
from TexSoup.tokens import Token
import zss


class SoupNode(object):

    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    @staticmethod
    def get_children(node: TexNode | Token) -> list[TexNode | Token]:
        if isinstance(node, TexNode):
            return node.contents
        return []

    @staticmethod
    def get_label(node: TexNode | Token) -> str:
        if isinstance(node, TexNode):
            return node.name
        return node


def TED_tikz(original, modified):
    """Tree edit distance, using the ZSS and TexSoup library
    Costs are 2 for addition/suppression of a node, and 1 for renaming

    Args:
        original (str): original LaTeX string
        modified (str): modified LaTeX string

    Returns:
        float: the TED between the AST of the original and modified code
    """
    return zss.distance(
        TexSoup(original),
        TexSoup(modified),
        SoupNode.get_children,
        lambda _: 2,
        lambda _: 2,
        lambda node1, node2: SoupNode.get_label(node1) != SoupNode.get_label(node2),
    )
