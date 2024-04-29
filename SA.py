import streamlit as st
import ast
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from io import BytesIO
import base64
import pandas as pd
import subprocess
import os

def perform_type_checking(source_code):
    temp_file_path = "temp.py"
    
    with open(temp_file_path, "w") as temp_file:
        temp_file.write(source_code)
    try:
        result = subprocess.run(["mypy", temp_file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return True, "Type checking passed successfully."
        else:
            return False, "Type Checking FAILED !!! Correct the code and analyze again."
    except Exception as e:
        return False, f"An error occurred during type checking: {e}"
    finally:
        os.remove(temp_file_path)
 
def main():
    st.set_page_config(page_title="CodeFlow", page_icon="👨‍💻")
    st.title("</> Code Flow 💻 - A tool for Control Flow 🔀 and Data Flow Analysis 📊")

    sample_sources = {
        "Sample 1": """
x = 1
y = 2
z = x + y
if z > 2:
    a = z - 1
else:
    a = z + 1
print(a)
""",
        "Sample 2": """
x = 5
while x > 0:
    y = x + 2
    x -= 1
print(y)
""",
        "Sample 3": """
x = 10
if x > 5:
    y = x + 2
else:
    y = 0
print(y)
"""
    }

    st.sidebar.header("COMPILER DESIGN PACKGE\n")
    selected_sample = st.sidebar.selectbox("Choose sample source code 👇", list(sample_sources.keys()))
    source_code = st.text_area("Enter your source code: ", sample_sources[selected_sample], height=250)
    st.sidebar.header("Tasks to perform ✔️❓")
    analyses_options = {
        "Control Flow Analysis": st.sidebar.checkbox("Control Flow Analysis"),
        "Reaching Definitions Analysis": st.sidebar.checkbox("Reaching Definitions Analysis"),
        "Live Variable Analysis": st.sidebar.checkbox("Live Variable Analysis"),
        "Constant Propagation Analysis": st.sidebar.checkbox("Constant Propagation Analysis"),
        "Variable Usage Graph": st.sidebar.checkbox("Variable Usage Graph"),
        "Loop Analysis": st.sidebar.checkbox("Loop Analysis")
    }
    #st.sidebar.text("--------------------------\n\nBy:\nPrakash\nVishal")

    if st.button("Analyze"):
        try:
            type_checking_passed, type_checking_message = perform_type_checking(source_code)
            if type_checking_passed:
                st.subheader("Type Checking process")
                st.success("Type checking passed!")
                ast_tree = ast.parse(source_code)
                st.subheader("Analyzing...")
                st.success("Lexical and syntax analysis successful.")
                optimized_ast = constant_folding(ast_tree)
                optimized_code = ast.unparse(optimized_ast)
                st.subheader("Optimized code by Constant Folding")
                st.code(optimized_code)
                cfg, entry_point = create_cfg(optimized_ast)

                if analyses_options["Control Flow Analysis"]:
                    st.subheader("Control Flow Analysis")
                    cfg_img = visualize_cfg(cfg, entry_point)
                    st.image(cfg_img, caption="Control Flow Graph")

                if any(analyses_options.values()):
                    dfa_results = perform_data_flow_analysis(cfg, optimized_ast)

                    if analyses_options["Reaching Definitions Analysis"]:
                        st.subheader("Reaching Definitions Analysis")
                        reaching_defs = dfa_results["Reaching Definitions Analysis"]
                        st.table(reaching_defs)

                    if analyses_options["Live Variable Analysis"]:
                        st.subheader("Live Variable Analysis")
                        live_vars = dfa_results["Live Variable Analysis"]
                        st.table(live_vars)

                    if analyses_options["Constant Propagation Analysis"]:
                        st.subheader("Constant Propagation Analysis")
                        constant_props = dfa_results["Constant Propagation Analysis"]
                        st.table(constant_props)

                    if analyses_options["Variable Usage Graph"]:
                        st.subheader("Variable Usage Graph")
                        variable_usage_img = visualize_variable_usage(cfg, dfa_results)
                        st.image(variable_usage_img, caption="Variable Usage Graph")

                    if analyses_options["Loop Analysis"]:
                        st.subheader("Loop Analysis")
                        loop_analysis_img = visualize_loops(cfg)
                        st.image(loop_analysis_img, caption="Loop Analysis")
            else:
                st.error(type_checking_message)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

def create_cfg(ast_tree):
    cfg = nx.DiGraph()
    entry_point = None

    class CFGVisitor(ast.NodeVisitor):
        def __init__(self, cfg):
            self.cfg = cfg
            self.current_node = None
            
        def generic_visit(self, node):
            node_id = f"{node.__class__.__name__}:{id(node)}"

            if node_id not in self.cfg:
                self.cfg.add_node(node_id, label=node.__class__.__name__)
            if self.current_node is not None:
                self.cfg.add_edge(self.current_node, node_id)
            previous_node = self.current_node
            self.current_node = node_id
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.AST):
                    self.generic_visit(child)
            self.current_node = previous_node
                
        def visit_If(self, node):
            self.visit(node.test)
            previous_node = self.current_node
            self.visit(node.body)
            
            if node.orelse:
                self.current_node = previous_node
                self.visit(node.orelse)
                
        def visit_While(self, node):
            self.visit(node.test)
            previous_node = self.current_node
            self.visit(node.body)

    visitor = CFGVisitor(cfg)
    visitor.visit(ast_tree)
    entry_point = list(cfg.nodes)[0]

    return cfg, entry_point


def visualize_cfg(cfg, entry_point):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(cfg)
    nx.draw(cfg, pos, with_labels=True, labels=nx.get_node_attributes(cfg, 'label'), node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    nx.draw_networkx_nodes(cfg, pos, nodelist=[entry_point], node_color='green', node_size=4000)
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_data}"

    return img_url

def constant_folding(ast_tree):    
    class ConstantFoldingVisitor(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                    try:
                        new_value = eval(compile(ast.Expression(node), '', 'eval'))
                        return ast.Constant(value=new_value, kind=None)
                    except ZeroDivisionError:
                        return node
            return node
        
        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.operand, ast.Constant):
                if isinstance(node.op, (ast.UAdd, ast.USub)):
                    new_value = eval(compile(ast.Expression(node), '', 'eval'))
                    return ast.Constant(value=new_value, kind=None)
            return node
        
        def visit_If(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant):
                if node.test.value:
                    return node.body
                else:
                    return node.orelse
            return node

    optimized_tree = ConstantFoldingVisitor().visit(ast_tree)
    ast.fix_missing_locations(optimized_tree)
    
    return optimized_tree

def perform_data_flow_analysis(cfg, ast_tree):

    dfa_results = {
        "Reaching Definitions Analysis": perform_reaching_definitions_analysis(cfg, ast_tree),
        "Live Variable Analysis": perform_live_variable_analysis(cfg, ast_tree),
        "Constant Propagation Analysis": perform_constant_propagation_analysis(cfg, ast_tree)
    }
 
    return dfa_results

def display_dfa_results(dfa_results):
    for node_id, value in dfa_results.items():
        st.write(f"Node {node_id}: {value}")

def perform_reaching_definitions_analysis(cfg, ast_tree):
    reaching_definitions = defaultdict(set)
    definitions = {}
       
    for node in ast.walk(ast_tree):
        node_id = f"{node.__class__.__name__}:{id(node)}"
        if isinstance(node, ast.Assign):       
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_var = target.id
                    definitions[node_id] = defined_var
                    reaching_definitions[node_id].add((defined_var, node_id))
    
        for successor in cfg.successors(node_id):
            reaching_definitions[successor].update(reaching_definitions[node_id])
    data = []

    for node_id, defs in reaching_definitions.items():
        node_class = node_id.split(":")[0]
        defs_list = ", ".join(f"({var}, {def_id})" for var, def_id in defs)
        data.append([node_class, node_id, defs_list])
    reaching_defs_df = pd.DataFrame(data, columns=['Node Class', 'Node ID', 'Reaching Definitions'])

    return reaching_defs_df

def perform_live_variable_analysis(cfg, ast_tree):
    live_variables = defaultdict(set)
    uses = defaultdict(set)
       
    for node in ast.walk(ast_tree):
        node_id = f"{node.__class__.__name__}:{id(node)}"
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name):
                uses[node_id].add(child.id)
    worklist = list(cfg.nodes)
    in_sets = defaultdict(set)
    out_sets = defaultdict(set)
    
    while worklist:
        current_node = worklist.pop(0)
        out_set = set()

        for succ in cfg.successors(current_node):
            out_set.update(in_sets[succ])
        in_set = uses[current_node] | (out_set - uses[current_node])
        
        if in_set != in_sets[current_node]:
            in_sets[current_node] = in_set
            worklist.extend(cfg.predecessors(current_node))
    data = []

    for node_id, live_vars in in_sets.items():
        node_class = node_id.split(":")[0]
        live_vars_list = ", ".join(live_vars)
        data.append([node_class, node_id, live_vars_list])
    live_vars_df = pd.DataFrame(data, columns=['Node Class', 'Node ID', 'Live Variables'])

    return live_vars_df

def perform_constant_propagation_analysis(cfg, ast_tree):
    constant_propagation = defaultdict(dict)
    constants = defaultdict(dict)

    for node in ast.walk(ast_tree):
        node_id = f"{node.__class__.__name__}:{id(node)}"
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        constants[node_id][var_name] = node.value.value
        for successor in cfg.successors(node_id):
            if node_id in constants:
                constant_propagation[successor].update(constants[node_id])
    data = []

    for node_id, consts in constant_propagation.items():
        node_class = node_id.split(":")[0]
        const_list = ", ".join(f"{var}: {value}" for var, value in consts.items())
        data.append([node_class, node_id, const_list])
    constant_props_df = pd.DataFrame(data, columns=['Node Class', 'Node ID', 'Constant Propagation'])

    return constant_props_df

def visualize_cfg_with_properties(cfg, entry_point, dfa_results):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(cfg)
    reaching_defs = dfa_results["Reaching Definitions Analysis"]
    live_vars = dfa_results["Live Variable Analysis"]
    constant_props = dfa_results["Constant Propagation Analysis"]
    nx.draw(cfg, pos, with_labels=True, labels=nx.get_node_attributes(cfg, 'label'), node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')

    for node in cfg.nodes:
        node_properties = []
        if node in reaching_defs:
            node_properties.append(f"Reaching Definitions: {reaching_defs[node]}")
        if node in live_vars:
            node_properties.append(f"Live Variables: {live_vars[node]}")
        if node in constant_props:
            node_properties.append(f"Constants: {constant_props[node]}")       
        label = "\n".join(node_properties)
        if label:
            plt.text(pos[node][0], pos[node][1] + 0.1, label, fontsize=8, ha='center', va='bottom')
    nx.draw_networkx_nodes(cfg, pos, nodelist=[entry_point], node_color='green', node_size=4000)

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)   
    img_data = base64.b64encode(img_buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_data}"

    return img_url

def visualize_variable_usage(cfg, dfa_results):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(cfg)   
    live_vars = dfa_results["Live Variable Analysis"]
    nx.draw(cfg, pos, with_labels=True, labels=nx.get_node_attributes(cfg, 'label'), node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    
    for node in cfg.nodes:
        if node in live_vars:       
            label = "\n".join(live_vars[node])
            plt.text(pos[node][0], pos[node][1] + 0.1, label, fontsize=8, ha='center', va='bottom')
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_data}"

    return img_url

def visualize_loops(cfg):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(cfg)  
    cycles = list(nx.simple_cycles(cfg))
    nx.draw(cfg, pos, with_labels=True, labels=nx.get_node_attributes(cfg, 'label'), node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    cycle_nodes = set()

    for cycle in cycles:
        cycle_nodes.update(cycle)
    nx.draw_networkx_nodes(cfg, pos, nodelist=cycle_nodes, node_color='orange', node_size=4000)
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    img_url = f"data:image/png;base64,{img_data}"

    return img_url

if __name__ == "__main__":
    main()
