function X = octane_data()
persistent data
if isempty(data)
    fid = fopen([fileparts(mfilename('fullpath')) '/x17.txt']);
    for a = 1:43
        fgets(fid);
    end
    data = textscan(fid, '%f64');
    fclose(fid);
    data = reshape(data{1}, 6, 82);
    data = data(2:end,:);
end
X = data;