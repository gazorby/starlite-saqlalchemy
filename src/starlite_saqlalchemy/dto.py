"""Using this implementation instead of the `starlite.SQLAlchemy` plugin DTO as
a POC for using the SQLAlchemy model type annotations to build the pydantic
model.

Also experimenting with marking columns for DTO purposes using the
`SQLAlchemy.Column.info` field, which allows demarcation of fields that
should always be private, or read-only at the model declaration layer.
"""
from __future__ import annotations

from collections import defaultdict
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
    from sqlalchemy.orm import Mapper, registry
    from sqlalchemy.sql.base import ReadOnlyColumnCollection
    from sqlalchemy.util import ReadOnlyProperties

from types import UnionType

AnyDeclarative = TypeVar("AnyDeclarative", bound=DeclarativeBase)


class DTOMapper:
    def __init__(self, registry: registry | None = None) -> None:
        self._mapped_classes: dict[str, type[DeclarativeBase]] = None
        self._registries: list[registry] = []
        if registry:
            self._registries.append(registry)
        self._model_modules: set[ModuleType] = set()

    @property
    def mapped_classes(self) -> dict[str, type[DeclarativeBase]]:
        if self._registries is None:
            from starlite_saqlalchemy.db.orm import Base

            self.add_registry(Base.registry)
        if self._mapped_classes is None:
            self._mapped_classes = {}
            for registry in self._registries:
                self._mapped_classes.update(
                    {m.class_.__name__: m.class_ for m in list(registry.mappers)}
                )
        return self._mapped_classes

    def clear_registries(self) -> None:
        self._registries = []
        self._mapped_classes = None

    def add_registry(self, registry: registry) -> None:
        self._registries.append(registry)

    def inspect_model(
        self, model: type[DeclarativeBase]
    ) -> tuple[ReadOnlyColumnCollection[str, Column], ReadOnlyProperties[RelationshipProperty]]:
        mapper = cast("Mapper", inspect(model))
        columns = mapper.columns
        relationships = mapper.relationships
        return columns, relationships

    def get_localns(self, model: type[DeclarativeBase]) -> dict[str, Any]:
        localns: dict[str, Any] = self.mapped_classes
        model_module = getmodule(model)
        if model_module is not None:
            self._model_modules.add(model_module)
        for module in self._model_modules:
            localns.update(vars(module))
        return localns

    def should_exclude_field(
        self,
        purpose: Purpose,
        elem: Column | RelationshipProperty,
        exclude: set[str],
        dto_attrib: Attrib,
    ) -> bool:
        if elem.key in exclude:
            return True
        if dto_attrib.mark is Mark.SKIP:
            return True
        if purpose is Purpose.WRITE and dto_attrib.mark is Mark.READ_ONLY:
            return True
        return False

    def construct_field_info(
        self, elem: Column | RelationshipProperty, purpose: Purpose
    ) -> FieldInfo:
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


class PydanticMapper(DTOMapper):
    supported_container_types = (List, list, Optional, Union, UnionType)

    def __init__(self, registry: registry | None = None) -> None:
        super().__init__(registry)
        self.dtos: dict[str, type[_MapperBind]] = {}
        self.dto_childs: dict[str, list[type[_MapperBind]]] = defaultdict(list)

    def _split_type(self, type_: type) -> tuple[list[type], dict[int, list[type]]]:
        outer_types, inner_types = [], []
        type_args = get_args(type_)
        type_origin = get_origin(type_)
        if type_origin is not None:
            outer_types.append(type_origin)
        elif not type_args:
            inner_types.append(type_)
        for arg in type_args:
            arg_outer_types, arg_inner_types = self._split_type(arg)
            outer_types.extend(arg_outer_types)
            inner_types.extend(arg_inner_types)
        return outer_types, inner_types

    def _build_union(self, union_types: tuple[Any, ...]) -> Any | Any:
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

    def _rebuild_type(self, outer_types: list[type], inner_types: list[type]) -> type:
        inner_types_ = inner_types
        if outer_types[-1] in (Union, UnionType):
            type_ = self._build_union(inner_types_)
        elif len(inner_types_) >= 1:
            type_ = outer_types[-1][inner_types_[0]]
        else:
            raise TypeError
        for outer_type in outer_types[-2::-1]:
            if outer_type not in self.supported_container_types:
                raise TypeError
            if outer_type in (Union, UnionType):
                if inner_types_ is None:
                    raise TypeError
                else:
                    type_ = self._build_union((type_, *inner_types_[1:]))
                    inner_types_ = None
            else:
                type_ = outer_type[type_]
        return type_

    def _is_model_ref(self, type_: Any) -> bool:
        return (isclass(type_) and issubclass(type_, DeclarativeBase)) or isinstance(
            type_, (ForwardRef, str)
        )

    def _ref_to_model(
        self, ref: type[DeclarativeBase] | ForwardRef | str
    ) -> type[DeclarativeBase] | None:
        if isinstance(ref, ForwardRef):
            return self.mapped_classes[ref.__forward_arg__]
        if isinstance(ref, str):
            return self.mapped_classes[ref]
        if isclass(ref) and issubclass(ref, DeclarativeBase):
            return ref
        return None

    def _resolve_type(
        self,
        name: str,
        type_: TypeAlias[Any] | ForwardRef,
        parents: dict[type[AnyDeclarative], str],
        purpose: Purpose,
        root: str,
        forward_refs: dict[type[AnyDeclarative], list[str]],
    ):
        outer_types, inner_types = self._split_type(type_)
        resolved_inner_types = []
        for inner_type in inner_types:
            model = self._ref_to_model(inner_type)
            if model is not None:
                dto_name = f"{name}_{model.__name__}"
                if model in parents:
                    dto = ForwardRef(dto_name)
                    forward_refs[model].append(dto_name)
                else:
                    dto = factory(
                        dto_name,
                        model,
                        purpose=purpose,
                        parents=parents,
                        root=root,
                        forward_refs=forward_refs,
                    )
                    self.dto_childs[root].append(dto)
                resolved_inner_types.append(dto)
            else:
                resolved_inner_types.append(inner_type)
        if outer_types and resolved_inner_types:
            return self._rebuild_type(outer_types, resolved_inner_types)
        elif len(resolved_inner_types) == 1:
            return resolved_inner_types[0]
        else:
            ValueError("fuck")

    def factory(
        self,
        name: str,
        model: type[AnyDeclarative],
        purpose: Purpose,
        *,
        exclude: set[str] | None = None,
        base: type[BaseModel] | None = None,
        parents: dict[type[AnyDeclarative], str] | None = None,
        forward_refs: dict[type[AnyDeclarative], list[str]] = None,
        root: str | None = None,
    ) -> type[_MapperBind[AnyDeclarative]]:

        if parents is None:
            parents = {}
        if root is None:
            root = name
        if forward_refs is None:
            forward_refs = defaultdict(list)
        parents[model] = name

        exclude = set() if exclude is None else exclude

        columns, relationships = self.inspect_model(model)
        fields: dict[str, tuple[Any, FieldInfo]] = {}
        validators: dict[str, AnyClassMethod] = {}
        for key, type_hint in get_type_hints(model, localns=self.get_localns(model)).items():
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

            if self.should_exclude_field(purpose, elem, exclude, attrib):
                continue

            if attrib.pydantic_type is not None:
                type_hint = attrib.pydantic_type

            for i, func in enumerate(attrib.validators or []):
                validators[f"_validates_{key}_{i}"] = validator(key, allow_reuse=True)(func)

            type_hint = self._resolve_type(name, type_hint, parents, purpose, root, forward_refs)

            fields[key] = (type_hint, self.construct_field_info(elem, purpose))

        dto = create_model(  # type:ignore[no-any-return,call-overload]
            name,
            __base__=tuple(filter(None, (base, _MapperBind[AnyDeclarative]))),
            __cls_kwargs__={"model": model},
            __module__=getattr(model, "__module__", __name__),
            __validators__=validators,
            **fields,
        )
        self.dtos[name] = dto

        if model_forward_refs := forward_refs.get(model, None):
            for forward_ref in model_forward_refs:
                self.dtos[forward_ref] = dto

        return dto


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

    @classmethod
    def update_forward_refs(cls, **localns: Any) -> None:
        namespace = {**pydantic_mapper.dtos, **localns}
        if childs := pydantic_mapper.dto_childs.get(cls.__name__):
            for child in childs:
                child.update_forward_refs(**namespace)
        else:
            return super().update_forward_refs(**namespace)


def _get_dto_attrib(elem: Column | RelationshipProperty) -> Attrib:
    return elem.info.get(settings.api.DTO_INFO_KEY, Attrib())


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


pydantic_mapper = PydanticMapper()
factory = pydantic_mapper.factory


def decorator(
    model: type[AnyDeclarative], purpose: Purpose, *, exclude: set[str] | None = None
) -> Callable[[type[BaseModel]], type[_MapperBind[AnyDeclarative]]]:
    """Infer a Pydantic model from SQLAlchemy model."""

    def wrapper(cls: type[BaseModel]) -> type[_MapperBind[AnyDeclarative]]:
        def wrapped() -> type[_MapperBind[AnyDeclarative]]:
            return pydantic_mapper.factory(cls.__name__, model, purpose, exclude=exclude, base=cls)

        return wrapped()

    return wrapper
