from pydantic import Field
from typing import List, Union, Literal, Optional
from sdks.novavision.src.base.model import (
    Package, Input, Output, Config,
    Inputs, Configs, Outputs, Response, Request, Detection,
)



class InputDistances(Input):
    """outputDistance from MeasureDistance — pairwise distance detections."""
    name: Literal["inputDistances"] = "inputDistances"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Distance Pairs"


class InputPersons(Input):
    """outputObjects from MeasureDistance — individual person detections (with trackerID if tracked)."""
    name: Literal["inputPersons"] = "inputPersons"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Persons (Pose Estimation)"


class InputFacialAnalysis(Input):
    """outputDetections from FacialAnalysis — persons enriched with age and gender."""
    name: Literal["inputFacialAnalysis"] = "inputFacialAnalysis"
    value: List[Detection]
    type: str = "list"

    class Config:
        title = "Facial Analysis"


class OutputGroups(Output):
    name: Literal["outputGroups"] = "outputGroups"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Social Groups"


class ProximityInputs(Inputs):
    inputDistances: InputDistances
    inputPersons: InputPersons
    inputFacialAnalysis: InputFacialAnalysis


class ProximityOutputs(Outputs):
    outputGroups: OutputGroups



class ConfigProximityThresholdCm(Config):
    """Maximum distance (cm) between two persons to be considered proximate."""
    name: Literal["configProximityThresholdCm"] = "configProximityThresholdCm"
    value: float = Field(default=150.0, ge=1.0, le=10000.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Proximity Distance Threshold (cm)"
        json_schema_extra = {"shortDescription": "Max Distance (cm)"}


class ConfigProximityDurationSec(Config):
    """Minimum seconds a pair must stay within threshold to form a stable group."""
    name: Literal["configProximityDurationSec"] = "configProximityDurationSec"
    value: float = Field(default=3.0, ge=0.0, le=3600.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Proximity Duration Threshold (sec)"
        json_schema_extra = {"shortDescription": "Min Duration (sec)"}



class GenderMaleFemale(Config):
    name: Literal["GenderMaleFemale"] = "GenderMaleFemale"
    value: Literal["both"] = "both"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Male + Female"


class GenderMale(Config):
    name: Literal["GenderMale"] = "GenderMale"
    value: Literal["male"] = "male"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Male"


class GenderFemale(Config):
    name: Literal["GenderFemale"] = "GenderFemale"
    value: Literal["female"] = "female"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Female"



class ConfigFamilyMinChildren(Config):
    """Minimum number of children required for Family classification."""
    name: Literal["configFamilyMinChildren"] = "configFamilyMinChildren"
    value: int = Field(default=1, ge=1, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Children Count"
        json_schema_extra = {"shortDescription": "Min. Children"}


class ConfigFamilyMinAdults(Config):
    """Minimum number of adults required for Family classification."""
    name: Literal["configFamilyMinAdults"] = "configFamilyMinAdults"
    value: int = Field(default=2, ge=1, le=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Adults Count"
        json_schema_extra = {"shortDescription": "Min. Adults"}


class ConfigFamilyAdultGender(Config):
    """Required adult gender combination for Family classification."""
    name: Literal["configFamilyAdultGender"] = "configFamilyAdultGender"
    value: Union[GenderMaleFemale, GenderMale, GenderFemale]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Required Adult Genders"
        json_schema_extra = {"shortDescription": "Gender Combination"}


class ConfigFamilyChildAge(Config):
    """Age threshold: persons younger than this are classified as children (Family)."""
    name: Literal["configFamilyChildAge"] = "configFamilyChildAge"
    value: int = Field(default=18, ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Child Age Threshold"
        json_schema_extra = {"shortDescription": "Age < X → Child"}


class ConfigFamilyHRValue(Config):
    """Bbox height ratio (max_height / person_height) above which a person is a child."""
    name: Literal["configFamilyHRValue"] = "configFamilyHRValue"
    value: float = Field(default=1.4, ge=1.0, le=5.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Height Ratio Value"
        json_schema_extra = {"shortDescription": "Ratio Threshold (e.g. 1.4)"}


class FamilyHREnabled(Config):
    name: Literal["FamilyHREnabled"] = "FamilyHREnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configFamilyHRValue: ConfigFamilyHRValue

    class Config:
        title = "Enabled"


class FamilyHRDisabled(Config):
    name: Literal["FamilyHRDisabled"] = "FamilyHRDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigFamilyHeightRatio(Config):
    """Fallback child detection using bbox height ratio when age data is unavailable."""
    name: Literal["configFamilyHeightRatio"] = "configFamilyHeightRatio"
    value: Union[FamilyHREnabled, FamilyHRDisabled]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Child Height Ratio (Fallback)"
        json_schema_extra = {"shortDescription": "Bbox Height Fallback"}


class FamilyEnabled(Config):
    name: Literal["FamilyEnabled"] = "FamilyEnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configFamilyMinChildren: ConfigFamilyMinChildren
    configFamilyMinAdults: ConfigFamilyMinAdults
    configFamilyAdultGender: ConfigFamilyAdultGender
    configFamilyChildAge: ConfigFamilyChildAge
    configFamilyHeightRatio: ConfigFamilyHeightRatio

    class Config:
        title = "Enabled"


class FamilyDisabled(Config):
    name: Literal["FamilyDisabled"] = "FamilyDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigFamilyDetection(Config):
    """Enable or disable Family group classification."""
    name: Literal["configFamilyDetection"] = "configFamilyDetection"
    value: Union[FamilyEnabled, FamilyDisabled]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Family Detection"
        json_schema_extra = {"shortDescription": "Classify Family Groups"}



class ConfigPCChildAge(Config):
    """Age threshold: persons younger than this are classified as children (Parent-Child)."""
    name: Literal["configPCChildAge"] = "configPCChildAge"
    value: int = Field(default=18, ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Child Age Threshold"
        json_schema_extra = {"shortDescription": "Age < X → Child"}


class ConfigPCHRValue(Config):
    """Bbox height ratio threshold for child detection fallback (Parent-Child)."""
    name: Literal["configPCHRValue"] = "configPCHRValue"
    value: float = Field(default=1.4, ge=1.0, le=5.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Height Ratio Value"
        json_schema_extra = {"shortDescription": "Ratio Threshold (e.g. 1.4)"}


class PCHREnabled(Config):
    name: Literal["PCHREnabled"] = "PCHREnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configPCHRValue: ConfigPCHRValue

    class Config:
        title = "Enabled"


class PCHRDisabled(Config):
    name: Literal["PCHRDisabled"] = "PCHRDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigPCHeightRatio(Config):
    """Fallback child detection using bbox height ratio when age data is unavailable."""
    name: Literal["configPCHeightRatio"] = "configPCHeightRatio"
    value: Union[PCHREnabled, PCHRDisabled]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Child Height Ratio (Fallback)"
        json_schema_extra = {"shortDescription": "Bbox Height Fallback"}


class ParentChildEnabled(Config):
    name: Literal["ParentChildEnabled"] = "ParentChildEnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configPCChildAge: ConfigPCChildAge
    configPCHeightRatio: ConfigPCHeightRatio

    class Config:
        title = "Enabled"


class ParentChildDisabled(Config):
    name: Literal["ParentChildDisabled"] = "ParentChildDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigParentChildDetection(Config):
    """Enable or disable Parent-Child group classification."""
    name: Literal["configParentChildDetection"] = "configParentChildDetection"
    value: Union[ParentChildEnabled, ParentChildDisabled]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Parent-Child Detection"
        json_schema_extra = {"shortDescription": "Classify Parent-Child Pairs"}



class ConfigCoupleMinDuration(Config):
    """Minimum seconds a pair must be proximate to be classified as a couple."""
    name: Literal["configCoupleMinDuration"] = "configCoupleMinDuration"
    value: float = Field(default=10.0, ge=0.0, le=3600.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Duration (sec)"
        json_schema_extra = {"shortDescription": "Min. Together Time"}


class FormationRequired(Config):
    name: Literal["FormationRequired"] = "FormationRequired"
    value: Literal["required"] = "required"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Yes — Side by Side Only"


class FormationNotRequired(Config):
    name: Literal["FormationNotRequired"] = "FormationNotRequired"
    value: Literal["not_required"] = "not_required"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "No — Any Formation"


class ConfigCoupleFormation(Config):
    """Whether side-by-side formation is required for couple classification."""
    name: Literal["configCoupleFormation"] = "configCoupleFormation"
    value: Union[FormationRequired, FormationNotRequired]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Formation Requirement"
        json_schema_extra = {"shortDescription": "Side-by-Side Required?"}


class CoupleGenderRequired(Config):
    name: Literal["CoupleGenderRequired"] = "CoupleGenderRequired"
    value: Literal["required"] = "required"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Male + Female Required"


class CoupleGenderAny(Config):
    name: Literal["CoupleGenderAny"] = "CoupleGenderAny"
    value: Literal["any"] = "any"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Any Gender"


class ConfigCoupleGender(Config):
    """Gender condition for couple classification."""
    name: Literal["configCoupleGender"] = "configCoupleGender"
    value: Union[CoupleGenderRequired, CoupleGenderAny]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Gender Condition"
        json_schema_extra = {"shortDescription": "Male+Female Required?"}


class CouplePoseEnabled(Config):
    name: Literal["CouplePoseEnabled"] = "CouplePoseEnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Enabled — Use Keypoints"


class CouplePoseDisabled(Config):
    name: Literal["CouplePoseDisabled"] = "CouplePoseDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigCouplePoseEstimation(Config):
    """Use COCO 17-keypoint pose estimation signals for couple classification."""
    name: Literal["configCouplePoseEstimation"] = "configCouplePoseEstimation"
    value: Union[CouplePoseEnabled, CouplePoseDisabled]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Pose Estimation"
        json_schema_extra = {"shortDescription": "Use Keypoints for Couple?"}


class CoupleEnabled(Config):
    name: Literal["CoupleEnabled"] = "CoupleEnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configCoupleMinDuration: ConfigCoupleMinDuration
    configCoupleFormation: ConfigCoupleFormation
    configCoupleGender: ConfigCoupleGender
    configCouplePoseEstimation: ConfigCouplePoseEstimation

    class Config:
        title = "Enabled"


class CoupleDisabled(Config):
    name: Literal["CoupleDisabled"] = "CoupleDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigCoupleDetection(Config):
    """Enable or disable Couple classification."""
    name: Literal["configCoupleDetection"] = "configCoupleDetection"
    value: Union[CoupleEnabled, CoupleDisabled]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Couple Detection"
        json_schema_extra = {"shortDescription": "Classify Couples"}



class ConfigFGMinSize(Config):
    """Minimum number of members to classify as a Friend Group."""
    name: Literal["configFGMinSize"] = "configFGMinSize"
    value: int = Field(default=2, ge=2, le=20)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Minimum Group Size"
        json_schema_extra = {"shortDescription": "Min. Members"}


class ConfigFGHeightTolerance(Config):
    """Max allowed bbox height ratio (tallest/shortest) within a friend group."""
    name: Literal["configFGHeightTolerance"] = "configFGHeightTolerance"
    value: float = Field(default=1.3, ge=1.0, le=5.0)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Height Tolerance Ratio"
        json_schema_extra = {"shortDescription": "Max Height Ratio (e.g. 1.3)"}


class ConfigFGChildAge(Config):
    """Age below which a person is considered a child for Friend Group child-skip logic."""
    name: Literal["configFGChildAge"] = "configFGChildAge"
    value: int = Field(default=18, ge=1, le=100)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Child Age Threshold"
        json_schema_extra = {"shortDescription": "Age < X → Child (skips group)"}


class FGChildSkip(Config):
    """Skip Friend Group classification if a child is detected in the group."""
    name: Literal["FGChildSkip"] = "FGChildSkip"
    value: Literal["skip"] = "skip"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configFGChildAge: ConfigFGChildAge

    class Config:
        title = "Skip If Child Present"


class FGChildAllow(Config):
    """Allow children in Friend Group — do not skip classification."""
    name: Literal["FGChildAllow"] = "FGChildAllow"
    value: Literal["allow"] = "allow"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Allow Children"


class ConfigFGChildTolerance(Config):
    """Whether to skip Friend Group classification when a child is present."""
    name: Literal["configFGChildTolerance"] = "configFGChildTolerance"
    value: Union[FGChildSkip, FGChildAllow]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Child Tolerance"
        json_schema_extra = {"shortDescription": "Behavior When Child Found"}


class FriendGroupEnabled(Config):
    name: Literal["FriendGroupEnabled"] = "FriendGroupEnabled"
    value: Literal["enabled"] = "enabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"
    configFGMinSize: ConfigFGMinSize
    configFGHeightTolerance: ConfigFGHeightTolerance
    configFGChildTolerance: ConfigFGChildTolerance

    class Config:
        title = "Enabled"


class FriendGroupDisabled(Config):
    name: Literal["FriendGroupDisabled"] = "FriendGroupDisabled"
    value: Literal["disabled"] = "disabled"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Disabled"


class ConfigFriendGroupDetection(Config):
    """Enable or disable Friend Group classification."""
    name: Literal["configFriendGroupDetection"] = "configFriendGroupDetection"
    value: Union[FriendGroupEnabled, FriendGroupDisabled]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Friend Group Detection"
        json_schema_extra = {"shortDescription": "Classify Friend Groups"}



class ProximityConfigs(Configs):
    configProximityThresholdCm: ConfigProximityThresholdCm
    configProximityDurationSec: ConfigProximityDurationSec
    configFamilyDetection: ConfigFamilyDetection
    configParentChildDetection: ConfigParentChildDetection
    configCoupleDetection: ConfigCoupleDetection
    configFriendGroupDetection: ConfigFriendGroupDetection


class ProximityRequest(Request):
    inputs: Optional[ProximityInputs] = None
    configs: ProximityConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ProximityResponse(Response):
    outputs: ProximityOutputs


class SocialGroup(Config):
    name: Literal["SocialGroup"] = "SocialGroup"
    value: Union[ProximityRequest, ProximityResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Proximity"
        json_schema_extra = {"target": {"value": 0}}


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[SocialGroup]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["Proximity"] = "Proximity"
