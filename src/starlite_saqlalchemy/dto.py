"""Using this implementation instead of the `starlite.SQLAlchemy` plugin DTO as
a POC for using the SQLAlchemy model type annotations to build the pydantic
model.

Also experimenting with marking columns for DTO purposes using the
`SQLAlchemy.Column.info` field, which allows demarcation of fields that
should always be private, or read-only at the model declaration layer.
"""
from __future__ import annotations

from enum import Enum, auto
from inspect import getmodule, isclass
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ForwardRef,
    Generic,
    List,
    NamedTuple,
    Optional,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, create_model, validator
from pydantic.fields import FieldInfo
from sqlalchemy import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, RelationshipProperty

from starlite_saqlalchemy import settings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pydantic.typing import AnyClassMethod
    from sqlalchemy import Column
    from sqlalchemy.orm import Mapper
    from sqlalchemy.sql.base import ReadOnlyColumnCollection
    from sqlalchemy.util import ReadOnlyProperties

from types import UnionType

AnyDeclarative = TypeVar("AnyDeclarative", bound=DeclarativeBase)

_GENERATED_DTO_MODELS: dict[str, type[_MapperBind]] = {}
_NAMESPACE_MODULES: set[ModuleType] = set()


class Mark(str, Enum):
    """For marking column definitions on the domain models.

    Example:
    ```python
    class Model(Base):
        ...
        updated_at: Mapped[datetime] = mapped_column(info={"dto": Mark.READ_ONLY})
    ```
    """

    READ_ONLY = "read-only"
    SKIP = "skip"


class Purpose(Enum):
    """For identifying the purpose of a DTO to the factory.

    The factory will exclude fields marked as private or read-only on the domain model depending
    on the purpose of the DTO.

    Example:
    ```python
    ReadDTO = dto.factory("AuthorReadDTO", Author, purpose=dto.Purpose.READ)
    ```
    """

    READ = auto()
    WRITE = auto()


class DTOInfo(TypedDict):
    """Represent dto infos suitable for info mapped_column infos param."""

    dto: Attrib


class Attrib(NamedTuple):
    """For configuring DTO behavior on SQLAlchemy model fields."""

    mark: Mark | None = None
    """Mark the field as read only, or skip."""
    pydantic_field: FieldInfo | None = None
    """If provided, used for the pydantic model for this attribute."""
    pydantic_type: Any | None = None
    """Override the field type on the pydantic model for this attribute."""
    validators: Iterable[Callable[[Any], Any]] | None = None
    """Single argument callables that are defined on the DTO as validators for the field."""


class _MapperBind(BaseModel, Generic[AnyDeclarative]):
    """Produce an SQLAlchemy instance with values from a pydantic model."""

    __sqla_model__: ClassVar[type[DeclarativeBase]]

    class Config:
        """Set orm_mode for `to_mapped()` method."""

        orm_mode = True

    def __init_subclass__(  # pylint: disable=arguments-differ
        cls, model: type[DeclarativeBase] | None = None, **kwargs: Any
    ) -> None:
        if model is not None:
            cls.__sqla_model__ = model
        super().__init_subclass__(**kwargs)

    def to_mapped(self) -> AnyDeclarative:
        """Create an instance of `self.__sqla_model__`

        Fill the bound SQLAlchemy model recursively with values from
        this dataclass.
        """
        as_model = {}
        for field in self.__fields__.values():
            value = getattr(self, field.name)
            if isinstance(value, (list, tuple)):
                value = [el.to_mapped() if isinstance(el, _MapperBind) else el for el in value]
            if isinstance(value, _MapperBind):
                value = value.to_mapped()
            as_model[field.name] = value
        return cast("AnyDeclarative", self.__sqla_model__(**as_model))


def _construct_field_info(elem: Column | RelationshipProperty, purpose: Purpose) -> FieldInfo:
    default = getattr(elem, "default", None)
    nullable = getattr(elem, "nullable", False)
    if purpose is Purpose.READ:
        return FieldInfo(...)
    if default is None:
        if nullable:
            return FieldInfo(default=None)
        if isinstance(elem, RelationshipProperty):
            if elem.uselist:
                return FieldInfo(default_factory=list)
            return FieldInfo(default=None)
        return FieldInfo(...)
    if default.is_scalar:
        return FieldInfo(default=default.arg)
    if default.is_callable:
        return FieldInfo(default_factory=lambda: default.arg({}))
    raise ValueError("Unexpected default type")


def _get_dto_attrib(elem: Column | RelationshipProperty) -> Attrib:
    return elem.info.get(settings.api.DTO_INFO_KEY, Attrib())


def _should_exclude_field(
    purpose: Purpose, elem: Column | RelationshipProperty, exclude: set[str], dto_attrib: Attrib
) -> bool:
    if elem.key in exclude:
        return True
    if dto_attrib.mark is Mark.SKIP:
        return True
    if purpose is Purpose.WRITE and dto_attrib.mark is Mark.READ_ONLY:
        return True
    return False


def _inspect_model(
    model: type[DeclarativeBase],
) -> tuple[ReadOnlyColumnCollection[str, Column], ReadOnlyProperties[RelationshipProperty]]:
    mapper = cast("Mapper", inspect(model))
    columns = mapper.columns
    relationships = mapper.relationships
    return columns, relationships


def _build_union(union_types: tuple[Any, ...]) -> Any | Any:
    """Build an Union type out of a type params.

    Python < 3.11 don't allow the `typing.Union[*params]` syntax
    so we build the union iteratively by pairs, using the following property :
    Union[a, b, c] = Union[Union[a, b], c]
    """
    if len(union_types) < 2:
        raise ValueError("At least two inner types are needed to build a Union")

    union: Any | Any = None
    for typ_a, typ_b in zip(union_types, union_types[1:]):
        union = Union[typ_a, typ_b] if union is None else Union[union, Union[typ_a, typ_b]]
    return union


def _is_model_ref(type_: Any) -> bool:
    return (isclass(type_) and issubclass(type_, DeclarativeBase)) or isinstance(type_, ForwardRef)


def _type_to_name(type_: type[DeclarativeBase] | ForwardRef) -> str:
    if isclass(type_) and issubclass(type_, DeclarativeBase):
        return type_.__name__
    if isinstance(type_, ForwardRef):
        return type_.__forward_arg__
    raise TypeError(f"can't resolve name of type {type_}")


def _resolve_type(
    name: str,
    type_: TypeAlias[Any] | ForwardRef,
    parents: dict[type[AnyDeclarative], str],
    purpose: Purpose,
) -> Any:
    type_args = get_args(type_)
    type_origin = get_origin(type_)

    if isclass(type_) and _is_model_ref(type_):
        if type_ in parents:
            type_ = ForwardRef(parents[type_])
        else:
            model_name = _type_to_name(type_)
            type_name = f"{name}_{model_name}"
            return factory(
                type_name,
                type_,
                purpose=purpose,
                model_name=model_name,
                parents=parents,
            )
    # list[model], List[mode] or Optional[model]
    if type_origin in (list, List, Optional) and _is_model_ref(type_args[0]):
        if type_args[0] in parents:
            type_ = ForwardRef(parents[type_args[0]])
        else:
            model_name = _type_to_name(type_args[0])
            type_name = f"{name}_{model_name}"
            type_ = factory(
                type_name,
                type_args[0],
                purpose=purpose,
                model_name=model_name,
                parents=parents,
            )
        if type_origin is Optional:
            return Optional[type_]
        return list[type_]  # type: ignore[valid-type]
    # model | None or  Union[model, None]
    # When using the new | optional syntax
    # field is typed as types.UnionType (instead of typing.Union)
    if type_origin in (Union, UnionType):
        models_in_union, others_in_union = [], []
        for arg in type_args:
            if not _is_model_ref(arg):
                others_in_union.append(arg)
                continue
            arg_model: ForwardRef | type[_MapperBind]
            if arg in parents:
                arg_model = ForwardRef(parents[arg])
            else:
                model_name = _type_to_name(arg)
                type_name = f"{name}_{model_name}"
                arg_model = factory(
                    type_name,
                    arg,
                    purpose=purpose,
                    model_name=model_name,
                    parents=parents,
                )
            models_in_union.append(arg_model)

        return _build_union((*models_in_union, *others_in_union))
    return type_


def _get_localns(model: type[DeclarativeBase]) -> dict[str, Any]:
    localns: dict[str, Any] = {}
    model_module = getmodule(model)
    if model_module is not None:
        _NAMESPACE_MODULES.add(model_module)
    for module in _NAMESPACE_MODULES:
        localns.update(vars(module))
    return localns


def mark(mark_type: Mark) -> DTOInfo:
    """Shortcut for ```python.

    {"dto": Attrib(mark=mark_type)}
    ```

    Example:

    ```python
    class User(DeclarativeBase):
        id: Mapped[UUID] = mapped_column(
            default=uuid4, primary_key=True, info=dto.mark(dto.Mark.READ_ONLY)
        )
        email: Mapped[str]
        password_hash: Mapped[str] = mapped_column(info=dto.mark(dto.Mark.SKIP))
    ```

    Args:
        mark_type: dto Mark

    Returns:
        A `DTOInfo` suitable to pass to `info` param of `mapped_column`
    """
    return {"dto": Attrib(mark=mark_type)}


def factory(
    name: str,
    model: type[AnyDeclarative],
    purpose: Purpose,
    *,
    exclude: set[str] | None = None,
    base: type[BaseModel] | None = None,
    model_name: str | None = None,
    parents: dict[type[AnyDeclarative], str] | None = None,
) -> type[_MapperBind[AnyDeclarative]]:
    """Infer a Pydantic model from a SQLAlchemy model.

    The fields that are included in the model can be controlled on the SQLAlchemy class
    definition by including a "dto" key in the `Column.info` mapping. For example:

    ```python
    class User(DeclarativeBase):
        id: Mapped[UUID] = mapped_column(
            default=uuid4, primary_key=True, info={"dto": Attrib(mark=dto.Mark.READ_ONLY)}
        )
        email: Mapped[str]
        password_hash: Mapped[str] = mapped_column(info={"dto": Attrib(mark=dto.Mark.SKIP)})
    ```

    In the above example, a DTO generated for `Purpose.READ` will include the `id` and `email`
    fields, while a model generated for `Purpose.WRITE` will only include a field for `email`.
    Notice that columns marked as `Mark.SKIP` will not have a field produced in any DTO object.

    Args:
        name: Name given to the DTO class.
        model: The SQLAlchemy model class.
        purpose: Is the DTO for write or read operations?
        exclude: Explicitly exclude attributes from the DTO.
        base: A subclass of `pydantic.BaseModel` to be used as the base class of the DTO.

    Returns:
        A Pydantic model that includes only fields that are appropriate to `purpose` and not in
        `exclude`.
    """
    model_name = model_name or name
    if model_name in _GENERATED_DTO_MODELS:
        return _GENERATED_DTO_MODELS[model_name]
    if parents is None:
        parents = {}
    parents[model] = name

    exclude = set() if exclude is None else exclude

    columns, relationships = _inspect_model(model)
    fields: dict[str, tuple[Any, FieldInfo]] = {}
    validators: dict[str, AnyClassMethod] = {}
    for key, type_hint in get_type_hints(model, localns=_get_localns(model)).items():
        # don't override fields that already exist on `base`.
        if base is not None and key in base.__fields__:
            continue

        if get_origin(type_hint) is Mapped:
            (type_hint,) = get_args(type_hint)

        elem: Column | RelationshipProperty
        if key in columns:
            elem = columns[key]
        elif key in relationships:
            elem = relationships[key]
        else:
            # class var, anything else??
            continue

        attrib = _get_dto_attrib(elem)

        if _should_exclude_field(purpose, elem, exclude, attrib):
            continue

        if attrib.pydantic_type is not None:
            type_hint = attrib.pydantic_type

        for i, func in enumerate(attrib.validators or []):
            validators[f"_validates_{key}_{i}"] = validator(key, allow_reuse=True)(func)

        type_hint = _resolve_type(name, type_hint, parents, purpose)

        fields[key] = (type_hint, _construct_field_info(elem, purpose))

    return create_model(  # type:ignore[no-any-return,call-overload]
        name,
        __base__=tuple(filter(None, (base, _MapperBind[AnyDeclarative]))),
        __cls_kwargs__={"model": model},
        __module__=getattr(model, "__module__", __name__),
        __validators__=validators,
        **fields,
    )


def decorator(
    model: type[AnyDeclarative], purpose: Purpose, *, exclude: set[str] | None = None
) -> Callable[[type[BaseModel]], type[_MapperBind[AnyDeclarative]]]:
    """Infer a Pydantic model from SQLAlchemy model."""

    def wrapper(cls: type[BaseModel]) -> type[_MapperBind[AnyDeclarative]]:
        def wrapped() -> type[_MapperBind[AnyDeclarative]]:
            return factory(cls.__name__, model, purpose, exclude=exclude, base=cls)

        return wrapped()

    return wrapper
