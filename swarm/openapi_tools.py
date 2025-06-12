# openapi_tools.py 

# 將 OpenAPI 規格轉為 Swarm-compatible 同步工具函式
# 您可以在任何地方這樣使用：
# from swarm.openapi_tools import load_tools_from_openapi
# 或者若您在 swarm/ 資料夾內部使用，也可以：
# from .openapi_tools import load_tools_from_openapi

import json, re, textwrap, requests
from typing import Any

__all__ = ["openapi_to_swarm_functions", "generate_swarm_function_code", "load_tools_from_openapi"]

def openapi_to_swarm_functions(openapi_spec):
    def resolve_ref(ref, components):
        ref_path = ref.lstrip("#/" ).split("/")
        schema = components
        for part in ref_path[1:]:
            schema = schema.get(part, {})
        return schema

    def _deduce_type(schema: dict) -> str:
        t = schema.get("type")
        if t:
            return t
        for key in ("anyOf", "oneOf"):
            for var in schema.get(key, []):
                vt = var.get("type")
                if vt in ("object", "array"):
                    return vt
        return "string"

    components = openapi_spec.get("components", {})
    functions = []

    for path, methods in openapi_spec.get("paths", {}).items():
        for method, details in methods.items():
            op_id      = details.get("operationId", f"{method}_{path}").replace("/", "_").replace("{", "").replace("}", "")
            summary    = details.get("summary", "")
            parameters = details.get("parameters", [])
            request_body = details.get("requestBody", {})

            props, required = {}, []

            for p in parameters:
                name, schema = p["name"], p.get("schema", {})
                props[name] = {
                    "type": _deduce_type(schema),
                    **({"description": p["description"]} if "description" in p else {}),
                    **({"example":     p["example"]}     if "example"     in p else {})
                }
                if p.get("required"):
                    required.append(name)

            if "content" in request_body:
                for content in request_body["content"].values():
                    schema = content.get("schema", {})
                    if "$ref" in schema:
                        schema = resolve_ref(schema["$ref"], components)

                    if schema.get("type") == "object" and schema.get("properties"):
                        for pn, pschema in schema["properties"].items():
                            props[pn] = {
                                "type": _deduce_type(pschema),
                                **({"description": pschema["description"]} if "description" in pschema else {}),
                                **({"example":     pschema["example"]}     if "example"     in pschema else {})
                            }
                        required.extend(schema.get("required", []))
                    else:
                        props["body"] = {
                            "type": "object",
                            "description": schema.get("description", "Request body"),
                        }
                        if (ex := schema.get("default") or schema.get("example")) is not None:
                            props["body"]["example"] = ex

            functions.append({
                "name": op_id,
                "description": summary,
                "method": method,
                "path": path,
                "properties": props,
                "required": list(set(required)) if required else []
            })

    return functions


def generate_swarm_function_code(fn_meta, base_url):
    name       = fn_meta["name"]
    desc       = fn_meta["description"]
    method     = fn_meta["method"]
    path       = fn_meta["path"]
    props      = fn_meta["properties"]
    required   = set(fn_meta["required"])

    url        = f"{base_url}{path}"
    path_keys  = re.findall(r"{(.*?)}", path)

    sig_parts = []
    for key, schema in props.items():
        typ = "list" if schema.get("type") == "array" else "dict" if schema.get("type") == "object" else "str"
        default = "" if key in required else " = None"
        sig_parts.append(f"{key}: {typ}{default}")
    sig = ", ".join(sig_parts)

    body_line = "json=body" if "body" in props else ""
    payload_line = f"params={{k: v for k, v in payload.items() if k not in {path_keys + ['body']} and v is not None}}"

    fn_code = f'''
def {name}({sig}) -> dict:
    """{desc}"""
    payload = locals().copy()
    import requests
    {body_line and "# Body payload"}
    resp = requests.{method}(f"{url}", {body_line}, {payload_line})
    resp.raise_for_status()
    return resp.json()
'''
    return textwrap.dedent(fn_code)


def load_tools_from_openapi(openapi_spec: dict, base_url: str) -> list:
    """根據 OpenAPI 規格與 base_url，直接 exec() 產生 Swarm 可用函式列表"""
    function_metas = openapi_to_swarm_functions(openapi_spec)
    tools = []
    for meta in function_metas:
        fn_code = generate_swarm_function_code(meta, base_url)
        exec(fn_code, globals())
        tools.append(globals()[meta["name"]])
    return tools
