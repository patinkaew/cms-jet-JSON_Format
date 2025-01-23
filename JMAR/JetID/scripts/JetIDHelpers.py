import correctionlib
import correctionlib.schemav2 as schema
import rich
import json
import numpy as np
import gzip

comparison_operators = [">=", "<=", ">", "<"]

variable_information_dict = {
    "eta": {
        "description": "pseudorapidity of the jet",
        "type": "real",
        "range": (-5.2, 5.2),
    },
    "abs_eta": {
        "description": "absolute value of pseudorapidity of the jet",
        "type": "real",
        "range": (0.0, 5.2),
    },
    "chHEF": {
        "description": "charged Hadron Energy Fraction",
        "type": "real",
        "range": (0.0, 1.0),
    },
    "neHEF": {
        "description": "neutral Hadron Energy Fraction",
        "type": "real",
        "range": (0.0, 1.0),
    },
    "chEmEF": {
        "description": "charged Electromagnetic Energy Fraction",
        "type": "real",
        "range": (0.0, 1.0),
    },
    "neEmEF": {
        "description": "neutral Electromagnetic Energy Fraction",
        "type": "real",
        "range": (0.0, 1.0),
    },
    "muEF": {
        "description": "muon Energy Fraction",
        "type": "real",
        "range": (0.0, 1.0),
    },
    "chMultiplicity": {
        "description": "charged Multiplicity",
        "type": "int",
        "range": (0, 99999), # different syntax for infinity in correctionlib <2.6.0 and >=2.6.0 so use 99999 for now
        "flow": "error",
    },
    "neMultiplicity": {
        "description": "neutral Multiplicity",
        "type": "int",
        "range": (0, 99999), # different syntax for infinity in correctionlib <2.6.0 and >=2.6.0 so use 99999 for now
        "flow": "error",
    },
    "multiplicity": {
        "description": "charged Multiplicity + neutral Multiplicity",
        "type": "int",
        "range": (0, 99999), # different syntax for infinity in correctionlib <2.6.0 and >=2.6.0 so use 99999 for now
        "flow": "error",
    },
}

# eta and abs_eta should not be used together
variable_ordering = ["eta", "abs_eta", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity", "multiplicity"]

legacy_to_nanoaod_name_map = {
    "CHF": "chHEF",
    "NHF": "neHEF",
    "CEMF": "chEmEF",
    "NEMF": "neEmEF",
    "MUF": "muEF",
    "CHM": "chMultiplicity",
    "NumNeutralParticle": "neMultiplicity",
    "NumConst": "multiplicity",
}

# flow option to use
flow_option = "clamp" # use clamp to handle open bound on upperbound of binning node

# change +- np.inf to "+inf"/"-inf" string
def inf2str(value):
    if np.isposinf(value):
        return "+inf"
    elif np.isneginf(value):
        return "-inf"
    return value

# split unit expression by comparison operator
# e.g. split "NHF>0.9" to "NHF", ">", "0.9"
def split_unit_expression(string_expression):
    var_name = None
    var_threshold_value = None
    for comparison_operator in comparison_operators:
        find_index = string_expression.find(comparison_operator)
        if find_index >= 0: # find
            var_name = string_expression[:find_index]
            var_threshold_value = float(string_expression[find_index+len(comparison_operator):])
            break
    if var_name is None:
        raise ValueError(f"Cannot parse comparison operators from {string_expression}. Available operators are " + ", ".join(comparison_operators))
    return var_name, comparison_operator, var_threshold_value

# remove extra white spaces and add spaces properly
def prettify_expression(string_expression):
    string_expression = string_expression.replace(" ", "") # remove spaces
    return " && ".join(string_expression.split("&&"))

# change legacy naming to branch names in nanoaod
def convert_legacy_to_nanoaod_name(string_expression):
    for miniaod_name, nanoaod_name in legacy_to_nanoaod_name_map.items():
        string_expression = string_expression.replace(miniaod_name, nanoaod_name)
    return string_expression

# sort unit expression (NHF>0.9) in full expression in specified order
def sort_expression(string_expression):
    var_name_to_expr_map = dict()
    for unit_str_expr in string_expression.split("&&"):
        var_name, comparison_operator, var_threshold_value = split_unit_expression(unit_str_expr.strip())
        var_name_to_expr_map[var_name] = unit_str_expr.strip()
    sorted_unit_str_expr = list()
    for name in variable_ordering:
        if name in var_name_to_expr_map:
            sorted_unit_str_expr.append(var_name_to_expr_map[name])
    return " && ".join([var_name_to_expr_map[name] for name in variable_ordering if name in var_name_to_expr_map])

# combine prettify, conversion to branch names in nanoaod, and sort
def preprocess_expression(string_expression):
    string_expression = prettify_expression(string_expression)
    string_expression = convert_legacy_to_nanoaod_name(string_expression)
    string_expression = sort_expression(string_expression)
    return string_expression

# create a correction for unit expression, e.g. NHF<0.9
def create_unit_jetId_correction(string_expression, valid_content=1):
    var_name, comparison_operator, var_threshold_value = split_unit_expression(string_expression)
    # bins are defined as [low, high)
    if variable_information_dict[var_name]["type"] == "int":
        if comparison_operator == ">":
            comparison_operator = ">="
            var_threshold_value += 1
        if comparison_operator == "<=":
            comparison_operator = "<"
            var_threshold_value += 1

    var_min_value, var_max_value = variable_information_dict[var_name]["range"]
    if var_threshold_value < var_min_value or var_threshold_value > var_max_value:
        raise ValueError(f"Threshold value must be in between allowed values ({var_min_value}, {var_max_value}) of a variable {var_name}") 
    content = [0.0, 0.0]
    if comparison_operator == "<" or comparison_operator == "<=":
        content = [valid_content, 0.0]
    elif comparison_operator == ">" or comparison_operator == ">=":
        content = [0.0, valid_content]

    return schema.Binning(
        nodetype = "binning",
        input = var_name,
        edges = [inf2str(var_min_value), var_threshold_value, inf2str(var_max_value)],
        flow = variable_information_dict[var_name].get("flow", flow_option),
        content = content,
    )

# create a JetID correction for one working point (e.g. Tight, TightLeptonVeto) and one era
def create_jetId_correction(eta_bin_edges, criteria_expressions,
                            input_abs_eta=False, # whether to use eta (False) or absolute eta (True) as input. If using eta (False), transform node to apply absolute value will be added 
                            name=None, description=None, version=1, verbose_description=False):
    assert len(eta_bin_edges)-1 == len(criteria_expressions), f"Mismatch number of eta bins ({len(eta_bin_edges)-1}) and criteria expression (len(criteria_expressions))"
    
    def create_jetId_correction_per_eta_bin(string_expression):
        string_expression = string_expression.replace(" ", "") # remove spaces
        inputs = list()
        content = 1
        for unit_str_expr in string_expression.split("&&")[::-1]:
            content = create_unit_jetId_correction(unit_str_expr, valid_content=content)
            inputs.append(content.input)
        return inputs, content
    
    criteria_inputs = list()
    if not input_abs_eta: # use eta as input
        criteria_inputs.append("eta")
    else: # use abs(eta) as input
        criteria_inputs.append("abs_eta")
    criteria_corrections = list()
    for expr in criteria_expressions:
        inputs, correction = create_jetId_correction_per_eta_bin(expr)
        criteria_inputs += inputs
        criteria_corrections.append(correction)
    input_names = [input_name for input_name in variable_ordering if input_name in criteria_inputs]
    inputs = [schema.Variable(name=input_name, type="real", description=variable_information_dict[input_name]["description"]) # need to use type="real"
              for input_name in input_names]
    if verbose_description:
        for eta_bin_idx in range(len(eta_bin_edges)-1):
            description += "\n" + f"abs(eta) in [{eta_bin_edges[eta_bin_idx]}, {eta_bin_edges[eta_bin_idx+1]}): " + criteria_expressions[eta_bin_idx]

    data = None
    if not input_abs_eta: # use eta as input
        data = schema.Transform(
            nodetype = "transform",
            input = "eta",
            rule = schema.Formula(
                nodetype = "formula",
                variables = ["eta"],
                parser = "TFormula",
                expression = "abs(x)",
            ),
            content = schema.Binning(
                nodetype = "binning",
                input = "eta",
                edges = eta_bin_edges,
                flow = variable_information_dict["eta"].get("flow", flow_option),
                content = criteria_corrections
            )
        )
    else: # use abs(eta) as input
        data = schema.Binning(
            nodetype = "binning",
            input = "abs(eta)",
            edges = eta_bin_edges,
            flow = variable_information_dict["abs_eta"].get("flow", flow_option),
            content = criteria_corrections
        )

    return schema.Correction(
        name = name,
        description = description,
        version = 1,
        inputs = inputs,
        output = schema.Variable(name="jet id", type="real", description="jet identification"),
        data = data
    )

# create a JetID correctionSet for multiple working point (e.g. Tight, TightLeptonVeto) and multiple eras
# by combining JetID corrections for each working point and each era
def create_jetId_correctionSet(criteria_tasks, description="", input_abs_eta=False):
    corrections = list()
    for criteria_task in criteria_tasks:
        criteria_task["criteria_expressions"] = [preprocess_expression(_) for _ in criteria_task["criteria_expressions"]]
        corrections.append(create_jetId_correction(**criteria_task, input_abs_eta=input_abs_eta))
    return schema.CorrectionSet(
        schema_version = 2,
        description = description,
        corrections = corrections,
    )

# write correctionSet to a json file
def write_correctionSet(correctionSet, filename, write_gzip=False):
    with open(filename, "w") as fout:
        json.dump(correctionSet.dict(exclude_unset=True), fout, indent=4)
    if write_gzip:
        with gzip.open(filename+".gz", "wt") as fout:
            json.dump(correctionSet.dict(exclude_unset=True), fout)
