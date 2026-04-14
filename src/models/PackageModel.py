from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package,
    Detection,
    Inputs,
    Configs,
    Outputs,
    Response,
    Request,
    Output,
    Input,
    Config,
)


class InputDetections(Input):
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection], Detection]
    type: str = "object"

    class Config:
        title = "Detections"


class InputDistances(Input):
    name: Literal["inputDistances"] = "inputDistances"
    value: Union[List[Detection], Detection]
    type: str = "object"

    class Config:
        title = "Distances"


class OptionAnyPair(Config):
    name: Literal["anyPair"] = "anyPair"
    value: Literal["any_pair"] = "any_pair"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Any Pair"


class OptionMixedGenderOnly(Config):
    name: Literal["mixedGenderOnly"] = "mixedGenderOnly"
    value: Literal["mixed_gender_only"] = "mixed_gender_only"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Mixed Gender Only"


class OptionFamily(Config):
    name: Literal["family"] = "family"
    value: Literal["family"] = "family"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Family"


class OptionCouple(Config):
    name: Literal["couple"] = "couple"
    value: Literal["couple"] = "couple"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Couple"


class OptionFriendGroup(Config):
    name: Literal["friendGroup"] = "friendGroup"
    value: Literal["friend_group"] = "friend_group"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Friend Group"


class OptionSoloShopper(Config):
    name: Literal["soloShopper"] = "soloShopper"
    value: Literal["solo_shopper"] = "solo_shopper"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Solo Shopper"


class OptionParentChild(Config):
    name: Literal["parentChild"] = "parentChild"
    value: Literal["parent:child"] = "parent:child"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Parent-Child"


class OptionAdultGroup(Config):
    name: Literal["adultGroup"] = "adultGroup"
    value: Literal["adult_group"] = "adult_group"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Adult Group"


class ConfigGroupDistance(Config):
    """Maximum distance in meters for grouping persons."""

    name: Literal["ConfigGroupDistance"] = "ConfigGroupDistance"
    value: float = Field(ge=0.5, le=4.0, default=2.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.5, 4.0]"] = "[0.5, 4.0]"

    class Config:
        title = "Group Distance (m)"
        json_schema_extra = {"shortDescription": "Max intra-group distance in meters"}


class ConfigCohesionWindow(Config):
    """Minimum observation duration to confirm group cohesion."""

    name: Literal["ConfigCohesionWindow"] = "ConfigCohesionWindow"
    value: int = Field(ge=10, le=300, default=30)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[10, 300]"] = "[10, 300]"

    class Config:
        title = "Cohesion Window (s)"
        json_schema_extra = {"shortDescription": "Min seconds to confirm group cohesion"}


class ConfigTrajectoryWeight(Config):
    """Weight of trajectory coherence in final decision."""

    name: Literal["ConfigTrajectoryWeight"] = "ConfigTrajectoryWeight"
    value: float = Field(ge=0.0, le=1.0, default=0.4)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Trajectory Weight"
        json_schema_extra = {"shortDescription": "Trajectory vs distance balance"}


class ConfigFamilyAgeGap(Config):
    """Minimum child-adult age gap for family classification."""

    name: Literal["ConfigFamilyAgeGap"] = "ConfigFamilyAgeGap"
    value: float = Field(ge=10.0, le=35.0, default=20.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[10, 35]"] = "[10, 35]"

    class Config:
        title = "Family Age Gap (years)"
        json_schema_extra = {"shortDescription": "Min age difference for family classification"}


class ConfigCoupleMode(Config):
    """Gender constraint for couple detection."""

    name: Literal["ConfigCoupleMode"] = "ConfigCoupleMode"
    value: Union[OptionAnyPair, OptionMixedGenderOnly]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Couple Mode"
        json_schema_extra = {"shortDescription": "Gender constraint for couple detection"}


class ConfigMinFriendGroupSize(Config):
    """Minimum size for friend group classification."""

    name: Literal["ConfigMinFriendGroupSize"] = "ConfigMinFriendGroupSize"
    value: int = Field(ge=3, le=10, default=3)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[3, 10]"] = "[3, 10]"

    class Config:
        title = "Min Friend Group Size"
        json_schema_extra = {"shortDescription": "Min persons for friend group classification"}


class ConfigConfidenceThreshold(Config):
    """Minimum demographic confidence for demographic rules."""

    name: Literal["ConfigConfidenceThreshold"] = "ConfigConfidenceThreshold"
    value: float = Field(ge=0.3, le=0.99, default=0.5)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.3, 0.99]"] = "[0.3, 0.99]"

    class Config:
        title = "Confidence Threshold"
        json_schema_extra = {"shortDescription": "Min demographic confidence to include in classification"}


class ConfigRelationTypes(Config):
    """Relation types to emit in the output."""

    name: Literal["ConfigRelationTypes"] = "ConfigRelationTypes"
    value: List[
        Union[
            OptionFamily,
            OptionCouple,
            OptionFriendGroup,
            OptionSoloShopper,
            OptionParentChild,
            OptionAdultGroup,
        ]
    ]
    type: Literal["object"] = "object"
    field: Literal["selectBox"] = "selectBox"

    class Config:
        title = "Enabled Relation Types"
        json_schema_extra = {"shortDescription": "Which group types to detect"}


class OutputGroups(Output):
    name: Literal["outputGroups"] = "outputGroups"
    value: Union[dict, list]
    type: str = "object"

    class Config:
        title = "Social Groups"


class OutputStats(Output):
    name: Literal["outputStats"] = "outputStats"
    value: Union[dict, list]
    type: str = "object"

    class Config:
        title = "Composition Stats"


class SocialGroupInputs(Inputs):
    inputDetections: InputDetections
    inputDistances: InputDistances


class SocialGroupConfigs(Configs):
    configGroupDistance: ConfigGroupDistance
    configCohesionWindow: ConfigCohesionWindow
    configTrajectoryWeight: ConfigTrajectoryWeight
    configFamilyAgeGap: ConfigFamilyAgeGap
    configCoupleMode: ConfigCoupleMode
    configMinFriendGroupSize: ConfigMinFriendGroupSize
    configConfidenceThreshold: ConfigConfidenceThreshold
    configRelationTypes: ConfigRelationTypes


class SocialGroupOutputs(Outputs):
    outputGroups: OutputGroups
    outputStats: OutputStats


class SocialGroupRequest(Request):
    inputs: Optional[SocialGroupInputs]
    configs: SocialGroupConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class SocialGroupResponse(Response):
    outputs: SocialGroupOutputs


class SocialGroupExecutor(Config):
    name: Literal["SocialGroup"] = "SocialGroup"
    value: Union[SocialGroupRequest, SocialGroupResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Social Group Classification"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    """Top-level proximity task selector."""

    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SocialGroupExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Proximity Task"
        json_schema_extra = {"target": "value", "shortDescription": "Select proximity analysis type"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["Proximity"] = "Proximity"
