UFUNCTION(BlueprintCallable, CustomThunk, Category = "ECamera|Utils", meta = (Latent, LatentInfo = "LatentInfo", WorldContext = "WorldContextObject", CustomStructureParam = "Value", ExpandEnumAsExecs = "OutPin", AdvancedDisplay = 4))
static void SetPropertyByName(const UObject* WorldContextObject, UObject* Object, FName PropertyName, const int32& Value, double Duration, TEnumAsByte<EEasingFunc::Type> Func, double Exp, FLatentActionInfo LatentInfo, ELatentOutputPins& OutPin);
DECLARE_FUNCTION(execSetPropertyByName)
{
    P_GET_OBJECT(UObject, WorldContextObject);
    P_GET_OBJECT(UObject, Object);
    P_GET_PROPERTY(FNameProperty, PropertyName);
    Stack.StepCompiledIn<FProperty>(NULL);
    void* ValuePtr = Stack.MostRecentPropertyAddress;
    FProperty* ValueProperty = Stack.MostRecentProperty;
    P_GET_PROPERTY(FDoubleProperty, Duration);
    P_GET_PROPERTY(FByteProperty, Func);
    P_GET_PROPERTY(FDoubleProperty, Exp);
    P_GET_STRUCT(FLatentActionInfo, LatentInfo);
    P_GET_ENUM_REF(ELatentOutputPins, OutPin);
    
    P_FINISH;
    P_NATIVE_BEGIN;
    auto [SrcProperty, SrcPtr] = GetNestedPropertyFromObject(Object, PropertyName);
    if (SrcProperty == nullptr || ValueProperty == nullptr)
    {
        return;
    }

    bool bSameType = SrcProperty->SameType(ValueProperty);
    bool bFloatType = SrcProperty->IsA<FFloatProperty>() && ValueProperty->IsA<FDoubleProperty>();
    if (bSameType || bFloatType)
    {
        if (UWorld* World = GEngine->GetWorldFromContextObject(WorldContextObject, EGetWorldErrorMode::LogAndReturnNull))
        {
            TArray<FESetPropertyLatentAction*>& ActionList = GetActionList<FESetPropertyLatentAction>();
            FESetPropertyLatentAction** ActionPtr = ActionList.FindByPredicate([SrcProperty = SrcProperty, SrcPtr = SrcPtr](FESetPropertyLatentAction* ThisAction) { return ThisAction->IsSameProperty(SrcProperty, SrcPtr); });
            FESetPropertyLatentAction* Action = ActionPtr == nullptr ? nullptr : *ActionPtr;
            if (Action != nullptr)
            {
                Action->SetInterrupt(true);
            }

            Action = new FESetPropertyLatentAction(Object, SrcProperty, SrcPtr, ValueProperty, ValuePtr, Duration, EEasingFunc::Type(Func), Exp, OutPin, LatentInfo);
            Action->OnActionCompletedOrInterrupted.AddLambda([&ActionList, &Action]() { ActionList.Remove(Action); });
            ActionList.Add(Action);
            World->GetLatentActionManager().AddNewAction(LatentInfo.CallbackTarget, LatentInfo.UUID, Action);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("The found property %s does not has the same type as given property %s, respectively are %s and %s"),
            *SrcProperty->NamePrivate.ToString(), *ValueProperty->NamePrivate.ToString(), *SrcProperty->GetCPPType(), *ValueProperty->GetCPPType());
    }
    
    P_NATIVE_END;
}

template<typename ActionType>
// @TODO: Error in Clang 16
//	requires std::derived_from<ActionType, class FPendingLatentAction>
static TArray<ActionType*>& GetActionList()
{
    static TArray<ActionType*> ActionList {};
    return ActionList;
}

static std::pair<FProperty*, void*> GetNestedPropertyFromObject(UObject* Object, FName PropertyName)
{
    if (!IsValid(Object))
    {
        UE_LOG(LogTemp, Warning, TEXT("Input Object is invalid when calling function GetNestedPropertyFromObject."));
        return std::make_pair(nullptr, nullptr);
    }
    
    return GetNestedPropertyFromObjectStruct(Object, Object->GetClass(), PropertyName.ToString());
}

static std::pair<FProperty*, void*> GetNestedPropertyFromObjectStruct(void* Object, UStruct* Struct, const FString& PropertyName /*@TODO: Should use FStringView to improve string efficiency. */)
{
	int FoundIndex;
	FString CurrentProperty;
	FString NextProperty;
	bool bFoundSeparator = PropertyName.FindChar('.', FoundIndex);
	
	if (bFoundSeparator)
	{
		CurrentProperty = PropertyName.Mid(0, FoundIndex);
		NextProperty = PropertyName.Mid(FoundIndex + 1, PropertyName.Len() - FoundIndex - 1);
	}
	else
	{
		CurrentProperty = PropertyName;
	}

	FProperty* Property = FindFProperty<FProperty>(Struct, FName(CurrentProperty));
	if (Property != nullptr)
	{
		void* Value = Property->ContainerPtrToValuePtr<void>(Object);

		if (NextProperty.IsEmpty())
		{
			if (Property->IsA<FNumericProperty>() || Property->IsA<FStructProperty>())
			{
				return std::make_pair(Property, Value);
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("Terminate property can only be numeric/struct type. Current type is %s."), *Property->GetClass()->GetName());
				return std::make_pair(nullptr, nullptr);
			}
		}
		else
		{
			const FStructProperty* PropAsStruct = CastField<FStructProperty>(Property);
			const FObjectProperty* PropAsObject = CastField<FObjectProperty>(Property);
			const FArrayProperty* PropAsArray = CastField<FArrayProperty>(Property);
			const FSetProperty* PropAsSet = CastField<FSetProperty>(Property);
			const FMapProperty* PropAsMap = CastField<FMapProperty>(Property);

			if (PropAsArray != nullptr || PropAsSet != nullptr || PropAsMap != nullptr)
			{
				UE_LOG(LogTemp, Warning, TEXT("Function GetNestedPropertyFromObjectStruct currently does not support container type."));
			}
			else if (PropAsStruct != nullptr)
			{
				return GetNestedPropertyFromObjectStruct(Value, PropAsStruct->Struct, NextProperty);
			}
			else if (PropAsObject != nullptr)
			{
				// Now Value points to the pointer that points to the real object. Must let it point to the object instead of the pointer. Ref: DiffUtils.cpp
				UObject* PropObject = *((UObject* const*)Value);
				return GetNestedPropertyFromObjectStruct(PropObject, PropObject->GetClass(), NextProperty);
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("Invalid property: %s. Non-terminal property can only be an object or struct."), *FString(CurrentProperty));
			}

			return std::make_pair(nullptr, nullptr);
		}
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("Cannot find property %s from UStruct %s."), *FString(CurrentProperty), *Struct->GetName());
		return std::make_pair(nullptr, nullptr);
	}
}