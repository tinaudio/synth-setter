import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.vst import param_specs

TABLE_PREAMBLE = """
\\begin{longtable}{rlccl}
 \\addlinespace[-\\aboverulesep]
 \\cmidrule[\\heavyrulewidth]{2-5}
    &
     Parameter &
     Simple &
     Full &
     Description \\\\\\cmidrule{2-5}"""

TABLE_POSTAMBLE = """
 \\cmidrule[\\heavyrulewidth]{2-5}
 \\addlinespace[-\\belowrulesep]
\\rule{0pt}{0ex}\\\\
\\caption{}
\\label{tab:}
\\end{longtable}"""

YES = "\\cmark"
NO = "\\xmark"


def clean_param(p: str):
    if p.startswith("a_"):
        p = p[2:]

    p = p.replace("fx_a1", "chorus")
    p = p.replace("fx_a2", "delay")
    p = p.replace("fx_a3", "reverb")

    # if p contains a number, replace the number with "i"
    for i in range(10):
        if "ring_" in p:
            continue
        p = p.replace(str(i), "i")

    p = p.replace("_", "\\_")

    return p


def main():
    simple_params = param_specs["surge_simple"].names
    full_params = param_specs["surge_xt"].names

    params = list(set(simple_params + full_params))
    params = sorted(params)
    params = [(p, True) if p in simple_params else (p, False) for p in params ]
    params = [(p, s, True) if p in full_params else (p, s, False) for p, s in params ]
    params = [(clean_param(p), s, f) for p, s, f in params]

    params = {p: {"simple": s, "full": f} for p, s, f in params}

    for k, v in params.items():
        if k in simple_params:
            v["simple"] = True
        if k in full_params:
            v["full"] = True

    print(TABLE_PREAMBLE)
    for k, v in params.items():
        print(
            f" &\n{k} &\n{YES if v['simple'] else NO} & {YES if v['full'] else NO} &\n \\\\"
        )

    print(TABLE_POSTAMBLE)


if __name__ == "__main__":
    main()
