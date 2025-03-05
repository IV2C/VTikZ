import re
def template_valid(template_code:str,prediction:str):
    exist_command:list[tuple[str,str,str]] = re.findall(r"§exist\(([^,]+),([^)]+)\)\{\|(.+?)\|\}", r"\1", template_code, flags=re.DOTALL)
    #remove the optional if exists
    #create a regex to find the pattern of the code inside the exist within the line args
    #if exists remove the pattern from both the template and prediction and iterate until either no more exist or pattern not found => return false
    # do re.match for any command:
        # get the index of the match(Match.start())
        # verify that the expression in the prediction at that index matches the command => yes replace with default in both template_code and prediction / no return false
            # §def(value) => replace in the rest of the entire prediction code the value defined in the prediction by the value of the template 
            # §range(low,hi,default) => find expression at index with regex,evaluate expression at index, then if >low <high replace with default else false
            # §rangei(value,interval) => find expression at index with regex,evaluate expression at index, then if >value-interval <value+interval replace with default else false
            # §choice([a,b,..],default)) => find expression at index with regex, not there false, there replace with default
            # §equal(exp) => evaluate expression at index , = exp is replace with exp
        # redo until no more command
    # once no more command strict equality between both codes
    re.match


def exist_valid(l1:int,l2:int,template_code:str):
    pass