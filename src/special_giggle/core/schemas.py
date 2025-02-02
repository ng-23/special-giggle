from marshmallow import fields, validate, Schema

registered_schemas = {}

def register_schema(name:str):
    def decorator(schema:Schema):
        registered_schemas[name] = schema
        return schema
    return decorator

@register_schema('OptunaSearchSpace')
class OptunaSearchSpace(Schema):
    '''
    Represents a hyperparameter search space for Optuna
    '''

    type = fields.String(required=True, validate=validate.OneOf(['float','int']))
    low = fields.Number(required=True)
    high = fields.Number(required=True)
    step = fields.Number(required=False, load_default=None)
    log = fields.Boolean(required=False, load_default=False)

def validate_config(schema_name:str, config:dict):
    if schema_name not in registered_schemas:
        raise Exception(f'Cannot validate supplied config, no such registered schema {schema_name} exists to validate against')
    schema = registered_schemas[schema_name]()

    return schema.load(config)