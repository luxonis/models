import json
import os
import re
import sys
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any, cast

from pydoctor.driver import get_system
from pydoctor.epydoc.markup import Field
from pydoctor.epydoc.markup._pyval_repr import colorize_inline_pyval, colorize_pyval
from pydoctor.epydoc.markup.google import get_parser
from pydoctor.model import (
    Attribute,
    Class,
    Documentable,
    DocumentableKind,
    Function,
    Options,
)

opts = Options.defaults()
opts.projectbasedirectory = Path(os.getcwd())
opts.sourcepath = [str(Path(os.getcwd()) / sys.argv[1])]
system = get_system(opts)


def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def serialize_annotation(annotation: str):
    return annotation.replace("<code>", "").replace("</code>", "")


def serialize_parameter(parameter: Parameter):
    data = {
        "name": parameter.name,
        "kind": str(parameter.kind),
    }

    if parameter.annotation is not Parameter.empty:
        data["type"] = colorize_inline_pyval(parameter.annotation).to_node().astext()
        data["type"] = remove_html_tags(data["type"])
    if parameter.default is not Parameter.empty:
        data["default"] = colorize_inline_pyval(parameter.default).to_node().astext()
        data["default"] = remove_html_tags(data["default"])

    return data


def serialize_attribute(obj, attr: Attribute):
    if attr.annotation is not None:
        obj["type"] = colorize_inline_pyval(attr.annotation).to_node().astext()

    if attr.value is not None:
        doc = colorize_pyval(
            attr.value,
            linelen=attr.system.options.pyvalreprlinelen,
            maxlines=attr.system.options.pyvalreprmaxlines,
        )

        obj["value"] = doc.to_node().astext()


def serialize_function(obj: dict[str, Any], func: Function) -> None:
    obj["is_async"] = func.is_async
    if func.signature is None:
        return

    obj["signature"] = {
        "parameters": list(map(serialize_parameter, func.signature.parameters.values()))
    }
    if func.signature.return_annotation is not Signature.empty:
        return_annotation = serialize_annotation(str(func.signature.return_annotation))
        pattern = r"(?:<a[^>]*>)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:</a>)?"
        match = re.search(pattern, return_annotation)
        if match:
            extracted_text = match.group("name")
            obj["signature"]["return_annotation"] = extracted_text  # type: ignore
        else:
            obj["signature"]["return_annotation"] = "None"  # type: ignore


def serialize_docstring_field(field: Field):
    try:
        obj = {
            "name": field.tag(),
            "body": field.body().to_node().astext(),
        }
    except NotImplementedError:
        return None
    arg = field.arg()
    if arg is not None:
        obj["arg"] = arg
    return obj


def build_json(json_arr, documentables: list[Documentable]):
    for doc in documentables:
        obj = {
            "name": doc.fullName(),
            "short_name": doc.name,
            "is_visible": doc.isVisible,
            "is_private": doc.isPrivate,
            "children": [],
        }
        if doc.kind is not None:
            obj["kind"] = doc.kind.name

        if (doc.parsed_docstring is None) and (doc.docstring is not None):
            # print(doc.docstring)
            parser = get_parser(doc)
            doc.parsed_docstring = parser(doc.docstring, [])

        if doc.parsed_docstring is not None:
            obj["docstring"] = {
                "fields": list(
                    filter(
                        lambda x: x is not None,
                        map(serialize_docstring_field, doc.parsed_docstring.fields),
                    )
                )
            }
            if doc.parsed_docstring.has_body:
                obj["docstring"]["summary"] = (  # type: ignore
                    doc.parsed_docstring.get_summary().to_node().astext()
                )
                obj["docstring"]["all"] = doc.parsed_docstring.to_node().astext()  # type: ignore

        if doc.parent is not None:
            obj["parent"] = doc.parent.fullName()

        build_json(obj["children"], list(doc.contents.values()))

        if doc.kind is DocumentableKind.CLASS:
            cls: Class = cast(Class, doc)
            obj["bases"] = cls.bases
        elif (
            doc.kind is DocumentableKind.FUNCTION or doc.kind is DocumentableKind.METHOD
        ):
            serialize_function(obj, cast(Function, doc))
        elif (
            doc.kind is DocumentableKind.ATTRIBUTE
            or doc.kind is DocumentableKind.CONSTANT
            or doc.kind is DocumentableKind.VARIABLE
            or doc.kind is DocumentableKind.TYPE_ALIAS
            or doc.kind is DocumentableKind.TYPE_VARIABLE
            or doc.kind is DocumentableKind.CLASS_VARIABLE
        ):
            serialize_attribute(obj, cast(Attribute, doc))

        json_arr.append(obj)


json_ready = []
build_json(json_ready, system.rootobjects)  # type: ignore

jsonified = json.dumps(json_ready)
with open("docs.json", "w") as f:
    f.write(jsonified)
