local Class = createClass{name="FileReaderWriter",bases={"base.Object"}};

--[[
Class: utils.FileReaderWriter

Reader class with support for CSV.

This class inherits from <base.Object>.
]]

--[=[
--[[
Constructor: FileReaderWriter

Create a new instance of the class.
]]
function FileReaderWriter(options)
]=]
function Class:initialize(options)

end

--[[
Function: readCSV

Helper method used to read a CSV file into a tensor
]]
function Class:readCSV(filename, nrows, ncols)
  -- Read data from CSV to tensor
  local csvFile = io.open(filename, 'r')  
  local header = csvFile:read()

  local data = torch.Tensor(nrows, ncols)

  local i = 0  
  for line in csvFile:lines('*l') do  
    i = i + 1
    local l = line:split(',')
    for key, val in ipairs(l) do
      data[i][key] = val
    end
  end

  csvFile:close()

  return data
end

--[[
Function: isUpdateRequired

Helper method used to check if a given file used as source
was updated (eg. modified) after a product file was generated
from that source
]]
function Class:isUpdateRequired(src,prod)
  local src_attr = lfs.attributes(src)
  local prod_attr = lfs.attributes(prod)

  return src_attr.modification > prod_attr.modification
end

--[[
Function: writeArray
]]
function Class:writeArray(filename,array)
  local file = io.open(filename,"w")
  for _,v in ipairs(array) do
    file:write(v.."\n")
  end
  file:close()
end

--[[
Function: writeTensor
]]
function Class:writeTensor(filename,tens)
  local ndim = tens:nDimension()

  local file = io.open(filename,"w")
  local nrows= tens:size(1)
  local sto = tens:storage()

  self:debug("Writing tensor file ",filename,"...")

  if ndim==1 then
    for i=1,nrows do 
      file:write(sto[1] .."\n")
    end
  elseif ndim==2 then
    local ncols = tens:size(2)
    for r=1,nrows do
      local tt = {}
      for c=1,ncols do
        table.insert(tt,tens[{r,c}])
      end
      file:write(table.concat(tt,",") .."\n")
    end
  else
    error("Cannot write tensor of ".. ndim.. " dimensions")
  end

  file:close()
  self:debug("Done.")
end

return Class


