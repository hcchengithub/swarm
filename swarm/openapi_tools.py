# openapi_tools.py 

# 將 OpenAPI 規格轉為 Swarm-compatible 同步工具函式
# 您可以在任何地方這樣使用：
# from swarm.openapi_tools import load_tools_from_openapi
# 或者若您在 swarm/ 資料夾內部使用，也可以：
# from .openapi_tools import load_tools_from_openapi

import json, re, textwrap, requests
from typing import Any

__all__ = ["openapi_to_swarm_functions", "generate_swarm_function_code", "load_tools_from_openapi"]

def enforce_no_additional(schema_obj: Any):
    """遞迴遍歷 schema 物件，若 type=object 則加上 additionalProperties=False"""
    if isinstance(schema_obj, dict):
        if schema_obj.get("type") == "object":
            schema_obj.setdefault("additionalProperties", False)
        for v in schema_obj.values():
            enforce_no_additional(v)
    elif isinstance(schema_obj, list):
        for item in schema_obj:
            enforce_no_additional(item)


def openapi_to_swarm_functions(openapi_spec):
    def resolve_ref(ref, components):
        ref_path = ref.lstrip("#/").split("/")
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
            op_id       = details.get("operationId", f"{method}_{path}")
            op_id       = op_id.replace("/", "_").replace("{", "").replace("}", "")
            summary     = details.get("summary", "")
            parameters  = details.get("parameters", [])
            request_body = details.get("requestBody", {})

            props, required = {}, []
            # 處理 path/query 參數
            for p in parameters:
                name, schema = p["name"], p.get("schema", {})
                prop = {
                    "type": _deduce_type(schema),
                    **({"description": p.get("description")} if p.get("description") else {}),
                    **({"example": p.get("example")}     if p.get("example")     else {})
                }
                props[name] = prop
                if p.get("required"):
                    required.append(name)

            # 處理 requestBody
            if "content" in request_body:
                for content in request_body["content"].values():
                    schema = content.get("schema", {})
                    if "$ref" in schema:
                        schema = resolve_ref(schema["$ref"], components)
                    if schema.get("type") == "object" and schema.get("properties"):
                        for pn, pschema in schema["properties"].items():
                            subprop = {
                                "type": _deduce_type(pschema),
                                **({"description": pschema.get("description")} if pschema.get("description") else {}),
                                **({"example": pschema.get("example")}     if pschema.get("example")     else {})
                            }
                            props[pn] = subprop
                        required.extend(schema.get("required", []))
                    else:
                        props["body"] = {
                            "type": "object",
                            "description": schema.get("description", "Request body"),
                        }
                        if (ex := schema.get("default") or schema.get("example")) is not None:
                            props["body"]["example"] = ex

            # 組裝 parameters schema
            param_schema = {
                "type": "object",
                "properties": props,
                "required": list(set(required)) if required else []
            }
            # 強制遞迴加上 additionalProperties=False
            enforce_no_additional(param_schema)

            functions.append({
                "name": op_id,
                "description": summary,
                "method": method,
                "path": path,
                "parameters": param_schema
            })

    return functions


def generate_swarm_function_code(fn_meta, base_url):
    name       = fn_meta["name"]
    desc       = fn_meta["description"]
    method     = fn_meta["method"]
    path       = fn_meta["path"]
    param_schema = fn_meta["parameters"]
    props      = param_schema["properties"]
    required   = set(param_schema.get("required", []))

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

    args_list = []
    if body_line:
        args_list.append(body_line)
    if payload_line:
        args_list.append(payload_line)
    joined_args = ", ".join(args_list)

    fn_code = f'''
def {name}({sig}) -> dict:
    """{desc}"""
    payload = locals().copy()
    import requests
    {body_line and "# Body payload"}
    resp = requests.{method}(f"{url}"{', ' + joined_args if joined_args else ''})
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
