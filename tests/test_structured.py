import pytest

from fastevals.structured import shorthand_to_schema


def test_all_scalar_aliases():
    schema = shorthand_to_schema("a:str,b:string,c:int,d:integer,e:float,f:number,g:bool,h:boolean,i:object,j:any")
    properties = schema["properties"]
    assert properties["a"] == {"type": "string"}
    assert properties["c"] == {"type": "integer"}
    assert properties["e"] == {"type": "number"}
    assert properties["g"] == {"type": "boolean"}
    assert properties["i"] == {"type": "object"}
    assert properties["j"] == {}
    assert schema["required"] == list(properties)


def test_optional_fields_are_not_required():
    schema = shorthand_to_schema("required:str,optional:str?,also_optional:int?")
    assert schema["required"] == ["required"]


def test_array_fields():
    schema = shorthand_to_schema("tags:str[],scores:float[]?,objects:object[]")
    assert schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert schema["properties"]["scores"] == {"type": "array", "items": {"type": "number"}}
    assert schema["required"] == ["tags", "objects"]


def test_descriptions_support_quotes_and_commas():
    schema = shorthand_to_schema(
        'id:str("Unique ID"),total:float("Amount, including taxes"),note:str(\'Optional note\')'
    )
    assert schema["properties"]["id"]["description"] == "Unique ID"
    assert schema["properties"]["total"]["description"] == "Amount, including taxes"
    assert schema["properties"]["note"]["description"] == "Optional note"


@pytest.mark.parametrize(
    "spec",
    ["", "field", ":str", "field:date", "field:str(", "field:str()"],
)
def test_invalid_specs_raise(spec):
    with pytest.raises(ValueError):
        shorthand_to_schema(spec)


def test_schema_is_closed_object():
    schema = shorthand_to_schema("name:str")
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False


def test_whitespace_is_ignored():
    schema = shorthand_to_schema('  name : str ( "A name" ) ,  age : int ?  ')
    assert schema["properties"]["name"] == {"type": "string", "description": "A name"}
    assert schema["required"] == ["name"]


def test_optional_marker_can_follow_description():
    schema = shorthand_to_schema('note:str("Optional note")?')
    assert "required" not in schema
    assert schema["properties"]["note"]["description"] == "Optional note"


def test_described_array():
    schema = shorthand_to_schema('tags:str[]("Classification tags")')
    assert schema["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "Classification tags",
    }


@pytest.mark.parametrize("spec", ["name:str,", ",name:str", "name:str,,age:int", "name:str,name:int"])
def test_empty_or_duplicate_fields_raise(spec):
    with pytest.raises(ValueError):
        shorthand_to_schema(spec)


@pytest.mark.parametrize("spec", ['name:str("unterminated)', "name:str(unquoted)", 'name:str("x"', 'name:str("x"))'])
def test_malformed_descriptions_raise(spec):
    with pytest.raises(ValueError):
        shorthand_to_schema(spec)


@pytest.mark.parametrize("spec", ["name:", "name:?", "name:str??", "name:str[][]", "name:DATE"])
def test_malformed_types_raise(spec):
    with pytest.raises(ValueError):
        shorthand_to_schema(spec)


def test_description_can_contain_parentheses_and_commas():
    schema = shorthand_to_schema('summary:str("Short summary (one sentence), if available")')
    assert schema["properties"]["summary"]["description"] == "Short summary (one sentence), if available"
