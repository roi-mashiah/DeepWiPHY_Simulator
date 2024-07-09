function tStruct = makeStructForConstant(data, rank, type)
%MAKESTRUCTFORCONSTANT Convert Constants to structs

%For tensor constants, data is dlarray labelled as 'U's
if isequal(type,"Tensor")
    data = dlarray(data,repmat('U', 1, max(2, rank)));
elseif isequal(type,"Integer")
    if isa(data,"dlarray")
        data = int64(extractdata(data));
    else
        data = int64(data);
    end
end
tStruct = struct("value",data,"rank",rank);

end
